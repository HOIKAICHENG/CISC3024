import random

import torch
from torch import nn
from torch.nn import functional as F


class BayesianBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, **kwargs)
        self.stochastic = True

    @classmethod
    def from_batchnorm(cls, source: nn.BatchNorm2d) -> "BayesianBatchNorm2d":
        reference = next(
            (
                tensor
                for tensor in (
                    source.weight,
                    source.bias,
                    source.running_mean,
                    source.running_var,
                )
                if tensor is not None
            ),
            None,
        )
        factory_kwargs = {}
        if reference is not None:
            factory_kwargs = {"device": reference.device, "dtype": reference.dtype}

        target = cls(
            source.num_features,
            eps=source.eps,
            momentum=source.momentum,
            affine=source.affine,
            track_running_stats=source.track_running_stats,
            **factory_kwargs,
        )
        target.load_state_dict(source.state_dict())
        target.train(source.training)
        if source.affine:
            target.weight.requires_grad_(source.weight.requires_grad)
            target.bias.requires_grad_(source.bias.requires_grad)
        return target

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.stochastic:
            return super().forward(input)

        self._check_input_dim(input)
        exponential_average_factor = (
            0.0 if self.momentum is None else self.momentum
        )
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )

        bn_training = self.training or (
            self.running_mean is None and self.running_var is None
        )
        running_mean = (
            self.running_mean
            if not self.training or self.track_running_stats
            else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )
        scale = self.weight
        if scale is None:
            scale = torch.ones(
                self.num_features, device=input.device, dtype=input.dtype
            )
        scale = scale * (1.0 + torch.randn_like(scale))
        return F.batch_norm(
            input,
            running_mean,
            running_var,
            scale,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


def convert_to_abnn(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(
            child, BayesianBatchNorm2d
        ):
            setattr(module, name, BayesianBatchNorm2d.from_batchnorm(child))
        else:
            convert_to_abnn(child)
    return module


def freeze_except_bayesian_norm(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, BayesianBatchNorm2d) and module.affine:
            module.weight.requires_grad = True
            module.bias.requires_grad = True


def set_stochastic(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, BayesianBatchNorm2d):
            module.stochastic = enabled


def class_weights_for_member(
    num_classes: int,
    emphasized: int,
    factor: float,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    selected = sorted(random.Random(seed).sample(range(num_classes), emphasized))
    weights = torch.ones(num_classes)
    weights[selected] = factor
    return weights, selected
