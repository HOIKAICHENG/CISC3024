"""ConvNeXt V2 backbone.

Re-implementation of the encoder described in
    Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders", CVPR 2023.

The only architectural difference w.r.t. ConvNeXt V1 lives in `Block`: the
LayerScale parameter is dropped and a Global Response Normalization (GRN)
layer is inserted after the GELU of the inverted bottleneck.  Both variants are
available through the `use_grn` flag so that the GRN ablation of the paper can
be reproduced.

The stem stride is configurable because CIFAR-10 images are 32x32 instead of
224x224; with `stem_stride=2` the four stages run at 16/8/4/2 resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_last (N, H, W, C) and channels_first (N, C, H, W)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"unsupported data_format {data_format}")
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization (Eq. 1-3 of the ConvNeXt V2 paper).

    Operates on channels_last tensors (N, H, W, C):
      1. aggregate   g_c = ||X_c||_2 over the spatial dimensions
      2. normalize   n_c = g_c / mean_c(g_c)              (divisive normalization)
      3. calibrate   X_c = gamma * (X_c * n_c) + beta + X_c
    gamma/beta start at zero so the layer is an identity at initialization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    """Stochastic depth applied per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    """ConvNeXt block. `use_grn=True` gives the V2 block, `False` the V1 block."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        use_grn: bool = True,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) if use_grn else None
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # V1 uses LayerScale; V2 replaces it by GRN.
        if not use_grn and layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    """Four-stage ConvNeXt V2 encoder with an optional classification head."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        stem_stride: int = 2,
        use_grn: bool = True,
    ):
        super().__init__()
        self.depths = list(depths)
        self.dims = list(dims)
        self.use_grn = use_grn

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stem_stride, stride=stem_stride),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(
                nn.Sequential(
                    *[
                        Block(dim=dims[i], drop_path=dp_rates[cur + j], use_grn=use_grn)
                        for j in range(depths[i])
                    ]
                )
            )
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if num_classes > 0:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Run the four stages.

        `mask` is a (N, h, w) tensor with 1 = masked.  Masked positions are
        zeroed after the stem and after every stage, which is the dense
        equivalent of the sparse convolutions used in the paper: it keeps
        masked regions from feeding information back into visible ones.
        """
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if mask is not None:
                x = x * (1.0 - self._resize_mask(mask, x.shape[-2:]))
            x = self.stages[i](x)
            if mask is not None:
                x = x * (1.0 - self._resize_mask(mask, x.shape[-2:]))
            feats.append(x)
        return x, feats

    @staticmethod
    def _resize_mask(mask: torch.Tensor, size) -> torch.Tensor:
        m = mask.unsqueeze(1).float()
        if m.shape[-2:] != tuple(size):
            m = F.interpolate(m, size=size, mode="nearest")
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.forward_features(x)
        x = self.norm(x.mean([-2, -1]))  # global average pooling
        return self.head(x)


def convnextv2_atto(num_classes: int = 10, use_grn: bool = True, **kwargs) -> ConvNeXtV2:
    return ConvNeXtV2(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), num_classes=num_classes, use_grn=use_grn, **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
