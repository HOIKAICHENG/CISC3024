import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from code.abnn import (
    BayesianBatchNorm2d,
    monte_carlo_probabilities,
    set_stochastic,
)


def test_monte_carlo_probabilities_are_normalized():
    models = [nn.Linear(2, 3), nn.Linear(2, 3)]
    loader = DataLoader(
        TensorDataset(torch.randn(5, 2), torch.arange(5) % 3), batch_size=2
    )
    probs, targets, sample_probs = monte_carlo_probabilities(
        models, loader, samples=2, device=torch.device("cpu")
    )
    assert probs.shape == (5, 3)
    assert sample_probs.shape == (4, 5, 3)
    assert targets.shape == (5,)
    assert np_all_finite(probs) and np_all_finite(sample_probs)
    assert torch.allclose(
        torch.from_numpy(probs.sum(1)), torch.ones(5), atol=1e-6
    )
    assert torch.allclose(
        torch.from_numpy(sample_probs.sum(-1)), torch.ones(4, 5), atol=1e-6
    )


def test_monte_carlo_probabilities_reject_changed_target_order():
    class ReorderedLoader:
        def __init__(self):
            self.passes = 0

        def __iter__(self):
            self.passes += 1
            inputs = torch.zeros(4, 2)
            labels = torch.arange(4)
            if self.passes > 1:
                labels = labels.flip(0)
            yield inputs[:2], labels[:2]
            yield inputs[2:], labels[2:]

    with pytest.raises(RuntimeError, match="loader order changed"):
        monte_carlo_probabilities(
            [nn.Linear(2, 3), nn.Linear(2, 3)],
            ReorderedLoader(),
            samples=1,
            device=torch.device("cpu"),
        )


def test_monte_carlo_samples_use_stochastic_normalization():
    class NoiseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = BayesianBatchNorm2d(2)
            self.fc = nn.Linear(2, 3)

        def forward(self, inputs):
            normalized = self.bn(inputs)
            return self.fc(normalized.mean(dim=(2, 3)))

    model = NoiseNet()
    set_stochastic(model, False)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 2, 3, 3), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    _, _, sample_probs = monte_carlo_probabilities(
        [model], loader, samples=2, device=torch.device("cpu")
    )
    assert not np.allclose(sample_probs[0], sample_probs[1])


def np_all_finite(values) -> bool:
    return bool(np.isfinite(values).all())


class _OODDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise AssertionError("OOD data was used during ABNN adaptation")


def _tiny_classifier():
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2, 10),
    )


def _adaptation_batch():
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(8, 1, 4, 4, generator=generator)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    return inputs, targets


def _loaders(shuffle: bool):
    inputs, targets = _adaptation_batch()
    train = DataLoader(
        TensorDataset(inputs, targets), batch_size=4, shuffle=shuffle
    )
    return {
        "train": train,
        "cifar10_test": train,
        "svhn_test": DataLoader(_OODDataset(), batch_size=1),
        "cifar100_test": DataLoader(_OODDataset(), batch_size=1),
    }


def _map_checkpoint(
    path,
    *,
    quick: bool = False,
    metadata=True,
    weight_offset: float = 0.0,
):
    from code.engine import save_training_state

    model = _tiny_classifier()
    with torch.no_grad():
        model[0].weight.add_(weight_offset)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_metadata = (
        {
            "stage": "map",
            "quick": quick,
            "epochs": 1 if quick else 20,
            "config": {},
        }
        if metadata
        else None
    )
    save_training_state(
        path,
        model,
        optimizer,
        epoch=0,
        history=[],
        metadata=checkpoint_metadata,
    )
    return path


def _config(tmp_path, **overrides):
    from code.config import ExperimentConfig

    values = {
        "member_seeds": (101, 102),
        "abnn_epochs": 2,
        "abnn_lr": 0.05,
        "momentum": 0.9,
        "emphasized_classes": 3,
        "emphasis_factor": 5.0,
        "num_workers": 0,
        "results_dir": str(tmp_path),
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_abnn_training_adapts_only_norm_on_training_data(tmp_path, monkeypatch):
    import code.train_abnn as train_abnn
    from code.abnn import BayesianBatchNorm2d
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt", quick=True)
    loaders = _loaders(shuffle=False)
    seen = []
    original = train_abnn.train_epoch

    def track_loader(model, loader, *args, **kwargs):
        seen.append(loader)
        return original(model, loader, *args, **kwargs)

    monkeypatch.setattr(train_abnn, "train_epoch", track_loader)
    config = _config(tmp_path / "quick", abnn_epochs=4, member_seeds=(101, 102, 103))

    members, histories = run_abnn_training(
        config, loaders, checkpoint, quick=True
    )

    assert len(members) == 1
    assert [row["epoch"] for row in histories[0]] == [1]
    assert seen == [loaders["train"]]
    trainable = [
        name
        for name, parameter in members[0].named_parameters()
        if parameter.requires_grad
    ]
    assert trainable == ["1.weight", "1.bias"]
    assert isinstance(members[0][1], BayesianBatchNorm2d)
    frozen_map = _tiny_classifier()
    from code.engine import load_training_state

    load_training_state(checkpoint, frozen_map)
    assert torch.equal(members[0][0].weight, frozen_map[0].weight)
    assert torch.equal(members[0][4].weight, frozen_map[4].weight)
    assert (tmp_path / "quick" / "abnn_member_0_quick.pt").exists()
    assert (tmp_path / "quick" / "abnn_history_quick.json").exists()
    assert not (tmp_path / "quick" / "abnn_member_0.pt").exists()
    assert not (tmp_path / "quick" / "abnn_history.json").exists()
    payload = json.loads(
        (tmp_path / "quick" / "abnn_history_quick.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["members"] == histories
    assert payload["config"]["abnn_epochs"] == 4
    assert payload["run"]["mode"] == "quick"
    assert payload["run"]["effective"]["abnn_epochs"] == 1
    assert payload["run"]["effective"]["member_seeds"] == [101]


def test_full_run_writes_one_artifact_per_member(tmp_path, monkeypatch):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt")
    config = _config(tmp_path / "full")

    members, histories = run_abnn_training(
        config, _loaders(shuffle=False), checkpoint, quick=False
    )

    assert len(members) == 2
    assert [len(history) for history in histories] == [2, 2]
    for index in range(2):
        assert (tmp_path / "full" / f"abnn_member_{index}.pt").exists()
        assert not (
            tmp_path / "full" / f"abnn_member_{index}_quick.pt"
        ).exists()
    assert (tmp_path / "full" / "abnn_history.json").exists()
    assert not (tmp_path / "full" / "abnn_history_quick.json").exists()


def test_member_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    map_source = _map_checkpoint(tmp_path / "map.pt")
    uninterrupted_map = uninterrupted_dir / "map.pt"
    resumed_map = resumed_dir / "map.pt"
    uninterrupted_dir.mkdir()
    resumed_dir.mkdir()
    uninterrupted_map.write_bytes(map_source.read_bytes())
    resumed_map.write_bytes(map_source.read_bytes())
    uninterrupted_config = _config(uninterrupted_dir)
    resumed_config = _config(resumed_dir)

    uninterrupted_members, uninterrupted_histories = run_abnn_training(
        uninterrupted_config, _loaders(shuffle=True), uninterrupted_map
    )

    original = train_abnn.train_epoch
    calls = 0

    def interrupt_before_final_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(train_abnn, "train_epoch", interrupt_before_final_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_abnn_training(
            resumed_config, _loaders(shuffle=True), resumed_map
        )
    assert calls == 4
    partial = torch.load(
        resumed_dir / "abnn_member_1.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert partial["epoch"] == 1

    resume_calls = 0

    def count_resume_epochs(*args, **kwargs):
        nonlocal resume_calls
        resume_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(train_abnn, "train_epoch", count_resume_epochs)
    resumed_members, resumed_histories = run_abnn_training(
        resumed_config, _loaders(shuffle=True), resumed_map
    )
    assert resume_calls == 1

    assert resumed_histories == uninterrupted_histories
    for resumed, uninterrupted in zip(resumed_members, uninterrupted_members):
        for left, right in zip(resumed.parameters(), uninterrupted.parameters()):
            assert torch.equal(left, right)


@pytest.mark.parametrize("stale_text", ['{"members": []}', "{broken json", None])
def test_completed_checkpoints_reconcile_history_without_retraining(
    tmp_path, monkeypatch, stale_text
):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt")
    config = _config(tmp_path)
    _, histories = run_abnn_training(
        config, _loaders(shuffle=False), checkpoint
    )
    history_path = tmp_path / "abnn_history.json"
    expected = json.loads(history_path.read_text(encoding="utf-8"))
    if stale_text is None:
        history_path.unlink()
    else:
        history_path.write_text(stale_text, encoding="utf-8")

    def fail_if_training_restarts(*args, **kwargs):
        pytest.fail("complete checkpoints unexpectedly restarted training")

    monkeypatch.setattr(train_abnn, "train_epoch", fail_if_training_restarts)
    _, resumed_histories = run_abnn_training(
        config, _loaders(shuffle=False), checkpoint
    )

    assert resumed_histories == histories
    assert json.loads(history_path.read_text(encoding="utf-8")) == expected


def test_adaptation_rejects_non_map_checkpoint(tmp_path, monkeypatch):
    import code.train_abnn as train_abnn
    from code.engine import save_training_state
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = tmp_path / "member.pt"
    model = _tiny_classifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_training_state(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        history=[{"epoch": 1, "loss": 0.5}],
        metadata={"stage": "abnn", "quick": False},
    )

    with pytest.raises(ValueError, match="MAP checkpoint"):
        run_abnn_training(
            _config(tmp_path), _loaders(shuffle=False), checkpoint
        )


def test_adaptation_rejects_map_checkpoint_without_metadata(
    tmp_path, monkeypatch
):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt", metadata=False)

    with pytest.raises(ValueError, match="required metadata"):
        run_abnn_training(
            _config(tmp_path), _loaders(shuffle=False), checkpoint
        )


@pytest.mark.parametrize(
    ("training_quick", "checkpoint_quick"),
    [(False, True), (True, False)],
)
def test_adaptation_rejects_quick_full_map_mismatch(
    tmp_path, monkeypatch, training_quick, checkpoint_quick
):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(
        tmp_path / "map.pt", quick=checkpoint_quick
    )

    with pytest.raises(ValueError, match="quick/full mode"):
        run_abnn_training(
            _config(tmp_path),
            _loaders(shuffle=False),
            checkpoint,
            quick=training_quick,
        )


def test_resume_rejects_members_from_replaced_map_checkpoint(
    tmp_path, monkeypatch
):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt")
    config = _config(tmp_path, abnn_epochs=1, member_seeds=(101,))
    run_abnn_training(config, _loaders(shuffle=False), checkpoint)
    member_state = torch.load(
        tmp_path / "abnn_member_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    original_digest = member_state["metadata"]["map_checkpoint_sha256"]
    assert original_digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    _map_checkpoint(checkpoint, weight_offset=1.0)

    with pytest.raises(ValueError, match="originating MAP checkpoint"):
        run_abnn_training(config, _loaders(shuffle=False), checkpoint)

    replaced_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert original_digest != replaced_digest


def test_resume_rejects_checkpoint_from_another_member(tmp_path, monkeypatch):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt")
    config = _config(tmp_path, abnn_epochs=1)
    run_abnn_training(config, _loaders(shuffle=False), checkpoint)
    member_one = tmp_path / "abnn_member_1.pt"
    member_one.write_bytes((tmp_path / "abnn_member_0.pt").read_bytes())

    with pytest.raises(ValueError, match="incompatible"):
        run_abnn_training(config, _loaders(shuffle=False), checkpoint)


@pytest.mark.parametrize(
    ("quick", "expected_members", "expected_samples", "expected_draws"),
    [(False, 3, 3, 9), (True, 1, 2, 2)],
)
def test_mode_policy_produces_expected_member_sample_draws(
    tmp_path,
    monkeypatch,
    quick,
    expected_members,
    expected_samples,
    expected_draws,
):
    import code.train_abnn as train_abnn
    from code.train_abnn import run_abnn_training

    monkeypatch.setattr(train_abnn, "resnet20", _tiny_classifier)
    checkpoint = _map_checkpoint(tmp_path / "map.pt", quick=quick)
    config = _config(
        tmp_path / "results",
        member_seeds=(101, 102, 103),
        abnn_epochs=1,
        mc_samples=3,
    )
    members, _ = run_abnn_training(
        config, _loaders(shuffle=False), checkpoint, quick=quick
    )
    samples = 2 if quick else config.mc_samples

    _, _, sample_probs = monte_carlo_probabilities(
        members,
        _loaders(shuffle=False)["cifar10_test"],
        samples=samples,
        device=torch.device("cpu"),
    )

    assert len(members) == expected_members
    assert samples == expected_samples
    assert sample_probs.shape[0] == expected_draws
