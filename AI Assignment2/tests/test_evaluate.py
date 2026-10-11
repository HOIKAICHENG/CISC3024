import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

import code.evaluate as evaluate
from code.config import ExperimentConfig
from code.data import CIFAR_MEAN, CIFAR_STD
from code.evaluate import evaluate_methods as _evaluate_methods
from code.metrics import classification_metrics, ood_metrics, predictive_entropy


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    mean = images.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    normalized = (images - mean) / std
    return DataLoader(TensorDataset(normalized, targets), batch_size=3)


def _model() -> nn.Module:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 2))
    with torch.no_grad():
        model[1].weight.copy_(
            torch.linspace(-0.4, 0.5, model[1].weight.numel()).reshape_as(
                model[1].weight
            )
        )
        model[1].bias.copy_(torch.tensor([-0.2, 0.3]))
    return model


def _provenance(member_count: int) -> dict:
    return {
        "map_checkpoint": {
            "path": "map_checkpoint.pt",
            "sha256": "a" * 64,
        },
        "member_checkpoints": [
            {
                "path": f"abnn_member_{index}.pt",
                "sha256": f"{index + 1:x}" * 64,
            }
            for index in range(member_count)
        ],
    }


def evaluate_methods(
    config,
    loaders,
    map_model,
    members,
    quick=False,
    *,
    provenance=None,
):
    return _evaluate_methods(
        config,
        loaders,
        map_model,
        members,
        quick=quick,
        provenance=provenance or _provenance(len(members)),
    )


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _model()
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.model(inputs)


def test_evaluation_is_a_self_consistent_persisted_artifact(tmp_path):
    images = torch.linspace(0.05, 0.95, 6 * 3 * 2 * 2).reshape(6, 3, 2, 2)
    targets = torch.tensor([0, 1, 0, 1, 1, 0])
    loaders = {
        "cifar10_test": _loader(images, targets),
        "svhn_test": _loader(images.flip(0), torch.tensor([1, 0, 1, 0, 0, 1])),
        "cifar100_test": _loader(
            images.roll(2, 0), torch.tensor([0, 0, 1, 1, 0, 1])
        ),
    }
    config = ExperimentConfig(
        results_dir=str(tmp_path), mc_samples=7, num_workers=0
    )

    result = evaluate_methods(
        config,
        loaders,
        _model(),
        [_model()],
        quick=True,
        provenance=_provenance(1),
    )

    assert result["run"]["mode"] == "quick"
    assert result["run"]["effective"]["member_count"] == 1
    assert result["run"]["effective"]["mc_samples"] == 2
    assert result["config"]["mc_samples"] == 7
    assert result["provenance"] == _provenance(1)
    assert set(result["methods"]) == {"map", "abnn_single", "abnn_full"}
    for method in result["methods"].values():
        probabilities = np.asarray(method["probabilities"])
        saved_targets = np.asarray(method["targets"])
        entropy = np.asarray(method["predictive_entropy"])
        assert method["classification"] == pytest.approx(
            classification_metrics(probabilities, saved_targets)
        )
        assert entropy == pytest.approx(predictive_entropy(probabilities))
        assert set(method["ood"]) == {"svhn", "cifar100"}
        assert set(method["ood_predictive_entropy"]) == {"svhn", "cifar100"}
        for dataset in ("svhn", "cifar100"):
            ood_entropy = np.asarray(
                method["ood_predictive_entropy"][dataset]
            )
            assert method["ood"][dataset] == pytest.approx(
                ood_metrics(entropy, ood_entropy)
            )

    assert np.asarray(result["example_images"]) == pytest.approx(
        images[:3].numpy()
    )
    assert result["example_targets"] == targets[:3].tolist()
    example_count = len(result["example_images"])
    for method in result["methods"].values():
        assert method["targets"][:example_count] == result["example_targets"]
        assert len(method["predictive_entropy"][:example_count]) == example_count

    serialized = json.dumps(result, allow_nan=False)
    saved = json.loads(
        (tmp_path / "evaluation_quick.json").read_text(encoding="utf-8")
    )
    assert saved == json.loads(serialized)
    assert not (tmp_path / "evaluation_quick.json.tmp").exists()


@pytest.mark.parametrize(
    ("quick", "mc_samples", "expected_forward_calls"),
    [
        (True, 9, [24]),
        (False, 5, [36, 18, 18]),
    ],
)
def test_evaluation_uses_approved_member_and_sample_policies(
    tmp_path, quick, mc_samples, expected_forward_calls
):
    images = torch.full((4, 3, 2, 2), 0.5)
    targets = torch.tensor([0, 1, 0, 1])
    loader = _loader(images, targets)
    member_count = 1 if quick else 3
    members = [_CountingModel() for _ in range(member_count)]
    evaluate_methods(
        ExperimentConfig(
            results_dir=str(tmp_path),
            mc_samples=mc_samples,
            num_workers=0,
        ),
        {
            "cifar10_test": loader,
            "svhn_test": loader,
            "cifar100_test": loader,
        },
        _model(),
        members,
        quick=quick,
    )

    assert [
        member.forward_calls for member in members
    ] == expected_forward_calls


@pytest.mark.parametrize(
    ("quick", "member_count", "expected_count"),
    [
        (True, 0, 1),
        (True, 2, 1),
        (False, 0, 3),
        (False, 1, 3),
        (False, 2, 3),
        (False, 4, 3),
    ],
)
def test_evaluation_rejects_wrong_member_count(
    tmp_path, quick, member_count, expected_count
):
    images = torch.full((4, 3, 2, 2), 0.5)
    targets = torch.tensor([0, 1, 0, 1])
    loader = _loader(images, targets)

    with pytest.raises(
        ValueError,
        match=rf"{'quick' if quick else 'full'} evaluation requires exactly "
        rf"{expected_count} ABNN member",
    ):
        evaluate_methods(
            ExperimentConfig(results_dir=str(tmp_path), num_workers=0),
            {
                "cifar10_test": loader,
                "svhn_test": loader,
                "cifar100_test": loader,
            },
            _model(),
            [_model() for _ in range(member_count)],
            quick=quick,
        )


@pytest.mark.parametrize(
    "mismatched_dataset", ["cifar10", "svhn", "cifar100"]
)
def test_evaluation_rejects_target_order_changes_for_every_dataset(
    tmp_path, monkeypatch, mismatched_dataset
):
    images = torch.full((4, 3, 2, 2), 0.5)
    loaders = {
        "cifar10_test": _loader(images, torch.tensor([0, 1, 0, 1])),
        "svhn_test": _loader(images, torch.tensor([1, 1, 0, 1])),
        "cifar100_test": _loader(images, torch.tensor([0, 0, 1, 1])),
    }
    dataset_names = {
        id(loader): key.removesuffix("_test")
        for key, loader in loaders.items()
    }
    expected_targets = {
        id(loader): torch.cat(
            [batch_targets for _, batch_targets in loader]
        ).numpy()
        for loader in loaders.values()
    }

    def deterministic_predictions(model, loader, device):
        targets = expected_targets[id(loader)]
        return np.full((len(targets), 2), 0.5), targets

    def reordered_monte_carlo(models, loader, samples, device):
        targets = expected_targets[id(loader)].copy()
        if dataset_names[id(loader)] == mismatched_dataset:
            targets = targets[::-1]
        probabilities = np.full((len(targets), 2), 0.5)
        return probabilities, targets, probabilities[None]

    monkeypatch.setattr(
        evaluate, "deterministic_probabilities", deterministic_predictions
    )
    monkeypatch.setattr(
        evaluate, "monte_carlo_probabilities", reordered_monte_carlo
    )

    with pytest.raises(
        RuntimeError, match=rf"{mismatched_dataset} target order differs"
    ):
        evaluate_methods(
            ExperimentConfig(results_dir=str(tmp_path), num_workers=0),
            loaders,
            _model(),
            [_model(), _model(), _model()],
        )


def test_evaluation_rejects_examples_not_aligned_with_saved_predictions(
    tmp_path, monkeypatch
):
    images = torch.full((4, 3, 2, 2), 0.5)
    targets = torch.tensor([0, 1, 0, 1])
    loader = _loader(images, targets)
    loaders = {
        "cifar10_test": loader,
        "svhn_test": loader,
        "cifar100_test": loader,
    }
    monkeypatch.setattr(
        evaluate,
        "_example_images",
        lambda current_loader: (images[:2].tolist(), [1, 0]),
    )

    with pytest.raises(RuntimeError, match="example target order differs"):
        evaluate_methods(
            ExperimentConfig(results_dir=str(tmp_path), num_workers=0),
            loaders,
            _model(),
            [_model(), _model(), _model()],
        )


def test_nonfinite_evaluation_does_not_replace_previous_artifact(
    tmp_path, monkeypatch
):
    output = tmp_path / "evaluation.json"
    previous = '{"status": "previous"}'
    output.write_text(previous, encoding="utf-8")
    images = torch.full((4, 3, 2, 2), 0.5)
    targets = torch.tensor([0, 1, 0, 1])
    loader = _loader(images, targets)
    monkeypatch.setattr(
        evaluate,
        "classification_metrics",
        lambda probabilities, saved_targets: {"accuracy": float("nan")},
    )

    with pytest.raises(ValueError, match="JSON compliant"):
        evaluate_methods(
            ExperimentConfig(results_dir=str(tmp_path), num_workers=0),
            {
                "cifar10_test": loader,
                "svhn_test": loader,
                "cifar100_test": loader,
            },
            _model(),
            [_model(), _model(), _model()],
        )

    assert output.read_text(encoding="utf-8") == previous
    assert not (tmp_path / "evaluation.json.tmp").exists()


def test_evaluation_rejects_randomized_dataset_order(tmp_path):
    images = torch.full((4, 3, 2, 2), 0.5)
    mean = images.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    loader = DataLoader(
        TensorDataset(
            (images - mean) / std,
            torch.zeros(4, dtype=torch.long),
        ),
        batch_size=2,
        shuffle=True,
    )

    with pytest.raises(ValueError, match="deterministic dataset order"):
        evaluate_methods(
            ExperimentConfig(results_dir=str(tmp_path), num_workers=0),
            {
                "cifar10_test": loader,
                "svhn_test": loader,
                "cifar100_test": loader,
            },
            _model(),
            [_model(), _model(), _model()],
        )


class _CustomSequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def test_evaluation_rejects_custom_sampler(tmp_path):
    images = torch.full((4, 3, 2, 2), 0.5)
    mean = images.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    dataset = TensorDataset(
        (images - mean) / std,
        torch.zeros(4, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=_CustomSequentialSampler(dataset),
    )

    with pytest.raises(ValueError, match="SequentialSampler"):
        evaluate_methods(
            ExperimentConfig(results_dir=str(tmp_path), num_workers=0),
            {
                "cifar10_test": loader,
                "svhn_test": loader,
                "cifar100_test": loader,
            },
            _model(),
            [_model(), _model(), _model()],
        )
