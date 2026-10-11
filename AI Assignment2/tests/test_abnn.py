import copy
import random

import torch
from torch import nn

from code.abnn import (
    BayesianBatchNorm2d,
    class_weights_for_member,
    convert_to_abnn,
    freeze_except_bayesian_norm,
    set_stochastic,
)


def test_conversion_preserves_state_prediction_and_freezes_backbone():
    source = nn.Sequential(
        nn.Conv2d(3, 4, 1, dtype=torch.float64),
        nn.Sequential(nn.BatchNorm2d(4, eps=1e-4, momentum=0.2, dtype=torch.float64)),
        nn.Flatten(),
    ).eval()
    with torch.no_grad():
        source[1][0].weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        source[1][0].bias.copy_(torch.tensor([-1.0, -0.5, 0.5, 1.0]))
        source[1][0].running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        source[1][0].running_var.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    x = torch.randn(2, 3, 5, 5, dtype=torch.float64)
    expected = source(x)

    model = copy.deepcopy(source)
    converted = convert_to_abnn(model)
    layer = converted[1][0]

    assert converted is model
    assert isinstance(layer, BayesianBatchNorm2d)
    assert not layer.training
    assert layer.eps == source[1][0].eps
    assert layer.momentum == source[1][0].momentum
    assert layer.weight.dtype == source[1][0].weight.dtype
    for name, value in source[1][0].state_dict().items():
        assert torch.equal(layer.state_dict()[name], value)

    set_stochastic(converted, False)
    assert torch.equal(converted(x), expected)

    freeze_except_bayesian_norm(converted)
    trainable = [
        name for name, parameter in converted.named_parameters() if parameter.requires_grad
    ]
    assert trainable == ["1.0.weight", "1.0.bias"]


def test_disabled_bayesian_layer_matches_batchnorm_training_running_stats():
    source = nn.BatchNorm2d(3, momentum=None, dtype=torch.float64)
    model = nn.Sequential(copy.deepcopy(source))
    converted = convert_to_abnn(model)
    set_stochastic(converted, False)

    for x in (
        torch.randn(4, 3, 3, 3, dtype=torch.float64),
        torch.randn(4, 3, 3, 3, dtype=torch.float64),
    ):
        assert torch.equal(converted(x), source(x))
        for name, value in source.state_dict().items():
            assert torch.equal(converted[0].state_dict()[name], value)


def test_conversion_supports_batchnorm_without_affine_or_running_stats():
    source = nn.Sequential(
        nn.BatchNorm2d(3, affine=False, track_running_stats=False)
    ).eval()
    x = torch.randn(2, 3, 4, 4)
    expected = source(x)

    converted = convert_to_abnn(copy.deepcopy(source))
    set_stochastic(converted, False)
    freeze_except_bayesian_norm(converted)

    assert torch.equal(converted(x), expected)
    assert list(converted.parameters()) == []


def test_enabled_bayesian_layer_changes_predictions():
    layer = BayesianBatchNorm2d(3).eval()
    x = torch.ones(2, 3, 4, 4)
    layer.stochastic = True

    assert not torch.equal(layer(x), layer(x))


def test_class_weights_are_deterministic_and_do_not_change_global_rng():
    random.seed(99)
    rng_state = random.getstate()
    expected_selected = sorted(random.Random(7).sample(range(10), 3))

    weights, selected = class_weights_for_member(10, 3, 5.0, seed=7)

    assert random.getstate() == rng_state
    assert selected == expected_selected
    assert torch.equal(
        weights,
        torch.tensor(
            [5.0 if index in expected_selected else 1.0 for index in range(10)]
        ),
    )
    repeated_weights, repeated_selected = class_weights_for_member(10, 3, 5.0, seed=7)
    assert torch.equal(repeated_weights, weights)
    assert repeated_selected == selected
