import random

import numpy as np
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


@torch.inference_mode()
def monte_carlo_probabilities(models, loader, samples: int, device):
    if samples < 1:
        raise ValueError("Monte Carlo sample count must be positive")
    if not models:
        raise ValueError("Monte Carlo prediction requires at least one model")
    draws = []
    saved_targets = None
    for model in models:
        model.eval()
        set_stochastic(model, True)
        for _ in range(samples):
            batches, targets = [], []
            for inputs, labels in loader:
                probabilities = model(inputs.to(device)).softmax(1).cpu()
                batches.append(probabilities)
                targets.append(labels.detach().cpu())
            draws.append(torch.cat(batches))
            current_targets = torch.cat(targets)
            if saved_targets is None:
                saved_targets = current_targets
            elif not torch.equal(saved_targets, current_targets):
                raise RuntimeError(
                    "evaluation loader order changed between Monte Carlo passes"
                )
    sample_probs = torch.stack(draws).numpy()
    probabilities = sample_probs.mean(axis=0)
    if not np.isfinite(sample_probs).all() or not np.isfinite(probabilities).all():
        raise ValueError("Monte Carlo probabilities must be finite")
    if np.any(sample_probs < 0) or np.any(probabilities < 0):
        raise ValueError("Monte Carlo probabilities must be non-negative")
    if not np.allclose(sample_probs.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("Monte Carlo sample probabilities must be normalized")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("Monte Carlo probabilities must be normalized")
    return probabilities, saved_targets.numpy(), sample_probs


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
