"""Fully Convolutional Masked AutoEncoder (FCMAE) for ConvNeXt V2.

Self-supervised pre-training task of Woo et al. (CVPR 2023): a random subset of
image patches is removed, the ConvNeXt encoder sees only the visible ones, and
a lightweight ConvNeXt decoder reconstructs the normalised pixels of the
removed patches.

Adaptation for 32x32 CIFAR-10 inputs: the paper masks at the resolution of the
last encoder stage (stride 32, a 7x7 grid for 224x224 inputs).  Here the last
stage is only 2x2, so the mask grid is decoupled from the encoder stride and
defined at 4x4 pixel units (an 8x8 grid, 64 mask units per image).  The decoder
upsamples the encoder output back to that grid before injecting mask tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from convnextv2 import Block, ConvNeXtV2, LayerNorm


class FCMAE(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        decoder_depth: int = 1,
        decoder_embed_dim: int = 128,
        mask_ratio: float = 0.6,
        norm_pix_loss: bool = True,
        stem_stride: int = 2,
        use_grn: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.grid = img_size // patch_size

        self.encoder = ConvNeXtV2(
            in_chans=in_chans,
            num_classes=0,
            depths=depths,
            dims=dims,
            stem_stride=stem_stride,
            use_grn=use_grn,
        )

        self.proj = nn.Conv2d(dims[-1], decoder_embed_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        self.decoder = nn.Sequential(
            *[Block(dim=decoder_embed_dim, use_grn=use_grn) for _ in range(decoder_depth)]
        )
        self.decoder_norm = LayerNorm(decoder_embed_dim, eps=1e-6, data_format="channels_first")
        self.pred = nn.Conv2d(decoder_embed_dim, patch_size**2 * in_chans, kernel_size=1)

        torch.nn.init.normal_(self.mask_token, std=0.02)
        for m in (self.proj, self.pred):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------ masks
    def gen_random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a (N, grid, grid) tensor, 1 = masked, with an exact mask ratio."""
        n = x.shape[0]
        num_patches = self.grid**2
        num_mask = int(num_patches * self.mask_ratio)
        noise = torch.rand(n, num_patches, device=x.device)
        ids = torch.argsort(noise, dim=1)
        mask = torch.zeros(n, num_patches, device=x.device)
        mask.scatter_(1, ids[:, :num_mask], 1.0)
        return mask.view(n, self.grid, self.grid)

    # -------------------------------------------------------------- patchify
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.in_chans
        h = w = self.grid
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p * p * c)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.in_chans
        h = w = self.grid
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # --------------------------------------------------------------- forward
    def forward_encoder(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = self.encoder.forward_features(imgs, mask=mask)
        return x

    def forward_decoder(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(latent)
        if x.shape[-1] != self.grid:
            x = F.interpolate(x, size=(self.grid, self.grid), mode="nearest")
        m = mask.unsqueeze(1).type_as(x)
        x = x * (1.0 - m) + self.mask_token.expand(x.shape[0], -1, self.grid, self.grid) * m
        x = self.decoder(x)
        x = self.decoder_norm(x)
        pred = self.pred(x)  # (N, p*p*C, grid, grid)
        return pred.flatten(2).transpose(1, 2)  # (N, grid*grid, p*p*C)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        m = mask.flatten(1)
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            mask = self.gen_random_mask(imgs)
        latent = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(latent, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
