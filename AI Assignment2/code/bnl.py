"""Bayesian Normalization Layer (BNL) from Franchi et al., CVPR 2024.

Replaces a pretrained BatchNorm2d with

    BNL(h) = ((h - mu) / sigma) * gamma * (1 + eps) + beta,
    eps ~ N(0, I)

so that the only extra stochasticity lives on the affine scale of an
already-trained normalisation layer.  Running statistics are copied from the
MAP checkpoint and then frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNL(nn.Module):
    """Drop-in replacement for ``nn.BatchNorm2d`` with multiplicative Gaussian noise on ``gamma``."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, noise_std: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.noise_std = noise_std
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.stochastic = True

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm2d, noise_std: float = 0.1) -> "BNL":
        layer = cls(bn.num_features, eps=bn.eps, momentum=bn.momentum, noise_std=noise_std)
        with torch.no_grad():
            if bn.weight is not None:
                layer.weight.copy_(bn.weight)
            if bn.bias is not None:
                layer.bias.copy_(bn.bias)
            layer.running_mean.copy_(bn.running_mean)
            layer.running_var.copy_(bn.running_var)
            layer.num_batches_tracked.copy_(bn.num_batches_tracked)
        return layer

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, noise_std={self.noise_std}, stochastic={self.stochastic}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always use the frozen MAP running statistics (paper: only gamma/beta are retrained).
        out = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            None,
            None,
            False,
            0.0,
            self.eps,
        )
        if self.stochastic:
            noise = self.noise_std * torch.randn(self.weight.shape, device=x.device, dtype=x.dtype)
            gamma = self.weight * (1.0 + noise)
        else:
            gamma = self.weight
        return out * gamma.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def replace_bn_with_bnl(module: nn.Module, noise_std: float = 0.1) -> nn.Module:
    """Recursively replace every ``BatchNorm2d`` with a BNL that inherits its state."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, BNL.from_batchnorm(child, noise_std=noise_std))
        else:
            replace_bn_with_bnl(child, noise_std=noise_std)
    return module


def set_bnl_stochastic(module: nn.Module, stochastic: bool) -> None:
    for m in module.modules():
        if isinstance(m, BNL):
            m.stochastic = stochastic


def freeze_except_bnl(module: nn.Module) -> list[nn.Parameter]:
    """Freeze the backbone; return the trainable BNL affine parameters."""
    for p in module.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, BNL):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            trainable.extend([m.weight, m.bias])
    return trainable


def count_bnl(module: nn.Module) -> int:
    return sum(isinstance(m, BNL) for m in module.modules())
