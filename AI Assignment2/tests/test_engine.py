import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from code.config import ExperimentConfig
from code.engine import (
    deterministic_probabilities,
    load_training_state,
    save_json,
    save_training_state,
    train_epoch,
)
from code.train_map import run_map_training


def test_train_epoch_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 2), torch.tensor([0, 1] * 4)),
        batch_size=4,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loss = train_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert loss > 0

    path = tmp_path / "state.pt"
    history = [{"epoch": 1, "loss": loss}]
    save_training_state(path, model, optimizer, epoch=1, history=history)

    restored = nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(
        restored.parameters(), lr=0.1, momentum=0.9
    )
    state = load_training_state(path, restored, restored_optimizer)

    assert state["epoch"] == 1
    assert state["history"] == history
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
    expected_optimizer = optimizer.state_dict()
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, expected_state in expected_optimizer["state"].items():
        for name, expected_value in expected_state.items():
            assert torch.equal(
                actual_optimizer["state"][parameter_id][name], expected_value
            )


def test_deterministic_probabilities_preserve_loader_order():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    inputs = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    probabilities, targets = deterministic_probabilities(
        model, loader, torch.device("cpu")
    )

    assert model.training is False
    assert probabilities.shape == (3, 2)
    assert torch.allclose(
        torch.from_numpy(probabilities).sum(dim=1), torch.ones(3)
    )
    assert targets.tolist() == labels.tolist()


def test_save_json_converts_numeric_and_path_values(tmp_path):
    path = tmp_path / "nested" / "history.json"

    save_json(
        path,
        {
            "loss": np.float32(0.25),
            "epochs": np.array([1, 2]),
            "accuracy": torch.tensor(0.75),
            "artifact": Path("results/checkpoint.pt"),
        },
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "loss": 0.25,
        "epochs": [1, 2],
        "accuracy": 0.75,
        "artifact": str(Path("results/checkpoint.pt")),
    }


def test_map_training_uses_warmup_then_cosine_and_writes_artifacts(
    tmp_path, monkeypatch
):
    import code.train_map as train_map

    monkeypatch.setattr(train_map, "resnet20", lambda: nn.Linear(2, 2))
    inputs = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [1.5, 0.5], [0.5, 1.5]]
    )
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    config = ExperimentConfig(
        map_epochs=4,
        map_lr=0.2,
        momentum=0.0,
        weight_decay=0.0,
        num_workers=0,
        results_dir=str(tmp_path),
    )

    model, history = run_map_training(
        config, {"train": loader, "cifar10_test": loader}
    )

    assert isinstance(model, nn.Linear)
    assert [row["epoch"] for row in history] == [1, 2, 3, 4]
    assert [row["lr"] for row in history] == pytest.approx(
        [0.02, 0.2, 0.15, 0.05]
    )
    history_payload = json.loads(
        (tmp_path / "map_history.json").read_text(encoding="utf-8")
    )
    assert history_payload["history"] == history
    assert history_payload["elapsed_seconds"] >= 0.0
    checkpoint = torch.load(
        tmp_path / "map_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 4


def test_map_resume_exactly_matches_uninterrupted_training(tmp_path, monkeypatch):
    import code.train_map as train_map

    def tiny_model():
        return nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))

    def loaders():
        generator = torch.Generator().manual_seed(91)
        inputs = torch.randn(20, 3, generator=generator)
        targets = (inputs[:, 0] + inputs[:, 1] > 0).long()
        return {
            "train": DataLoader(
                TensorDataset(inputs, targets), batch_size=4, shuffle=True
            ),
            "cifar10_test": DataLoader(
                TensorDataset(inputs, targets), batch_size=5
            ),
        }

    monkeypatch.setattr(train_map, "resnet20", tiny_model)
    common = {
        "map_epochs": 4,
        "map_lr": 0.05,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "num_workers": 0,
    }
    uninterrupted_config = ExperimentConfig(
        **common, results_dir=str(tmp_path / "uninterrupted")
    )
    resumed_config = ExperimentConfig(
        **common, results_dir=str(tmp_path / "resumed")
    )

    uninterrupted_model, uninterrupted_history = run_map_training(
        uninterrupted_config, loaders()
    )

    original_train_epoch = train_map.train_epoch
    calls = 0

    def interrupt_before_third_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_train_epoch(*args, **kwargs)

    monkeypatch.setattr(
        train_map, "train_epoch", interrupt_before_third_epoch
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_map_training(resumed_config, loaders())
    interrupted_state = torch.load(
        tmp_path / "resumed" / "map_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert interrupted_state["epoch"] == 2

    monkeypatch.setattr(train_map, "train_epoch", original_train_epoch)
    resumed_model, resumed_history = run_map_training(resumed_config, loaders())

    assert resumed_history == uninterrupted_history
    for name, value in uninterrupted_model.state_dict().items():
        assert torch.equal(resumed_model.state_dict()[name], value)


def test_map_resume_rejects_changed_training_configuration(
    tmp_path, monkeypatch
):
    import code.train_map as train_map

    monkeypatch.setattr(train_map, "resnet20", lambda: nn.Linear(2, 2))
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    loaders = {"train": loader, "cifar10_test": loader}
    original = ExperimentConfig(
        map_epochs=1,
        map_lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        num_workers=0,
        results_dir=str(tmp_path),
    )
    changed = ExperimentConfig(
        map_epochs=1,
        map_lr=0.2,
        momentum=0.0,
        weight_decay=0.0,
        num_workers=0,
        results_dir=str(tmp_path),
    )
    run_map_training(original, loaders)

    with pytest.raises(ValueError, match="incompatible"):
        run_map_training(changed, loaders)


def test_map_resume_rejects_epoch_beyond_target(tmp_path, monkeypatch):
    import code.train_map as train_map

    monkeypatch.setattr(train_map, "resnet20", lambda: nn.Linear(2, 2))
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    loaders = {"train": loader, "cifar10_test": loader}
    config = ExperimentConfig(
        map_epochs=1,
        momentum=0.0,
        weight_decay=0.0,
        num_workers=0,
        results_dir=str(tmp_path),
    )
    run_map_training(config, loaders)
    checkpoint_path = tmp_path / "map_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint["epoch"] = 2
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="exceeds target"):
        run_map_training(config, loaders)


def test_map_resume_accumulates_elapsed_time(tmp_path, monkeypatch):
    import code.train_map as train_map

    monkeypatch.setattr(train_map, "resnet20", lambda: nn.Linear(2, 2))
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
    )
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    loaders = {"train": loader, "cifar10_test": loader}
    config = ExperimentConfig(
        map_epochs=3,
        momentum=0.0,
        weight_decay=0.0,
        num_workers=0,
        results_dir=str(tmp_path),
    )
    original_train_epoch = train_map.train_epoch
    calls = 0

    def interrupt_before_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_train_epoch(*args, **kwargs)

    clock = iter([10.0, 12.0, 20.0, 22.0, 25.0])
    monkeypatch.setattr(train_map.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        train_map, "train_epoch", interrupt_before_second_epoch
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_map_training(config, loaders)
    first_payload = json.loads(
        (tmp_path / "map_history.json").read_text(encoding="utf-8")
    )
    assert first_payload["elapsed_seconds"] == 2.0

    monkeypatch.setattr(train_map, "train_epoch", original_train_epoch)
    run_map_training(config, loaders)
    final_payload = json.loads(
        (tmp_path / "map_history.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        tmp_path / "map_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert final_payload["elapsed_seconds"] == 7.0
    assert checkpoint["elapsed_seconds"] == 7.0


def test_completed_quick_checkpoint_recreates_missing_history(
    tmp_path, monkeypatch
):
    import code.train_map as train_map

    monkeypatch.setattr(train_map, "resnet20", lambda: nn.Linear(2, 2))
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    loaders = {"train": loader, "cifar10_test": loader}
    config = ExperimentConfig(
        map_epochs=4,
        momentum=0.0,
        weight_decay=0.0,
        num_workers=0,
        results_dir=str(tmp_path),
    )

    _, history = run_map_training(config, loaders, quick=True)
    checkpoint_path = tmp_path / "map_checkpoint_quick.pt"
    history_path = tmp_path / "map_history_quick.json"
    assert checkpoint_path.exists()
    assert len(history) == 1
    history_path.unlink()

    _, resumed_history = run_map_training(config, loaders, quick=True)

    assert resumed_history == history
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert payload["history"] == history


def test_map_training_reports_corrupt_checkpoint(tmp_path, monkeypatch):
    import code.train_map as train_map

    monkeypatch.setattr(train_map, "resnet20", lambda: nn.Linear(2, 2))
    (tmp_path / "map_checkpoint.pt").write_bytes(b"not a torch checkpoint")
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    config = ExperimentConfig(
        map_epochs=1,
        num_workers=0,
        results_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="could not load checkpoint"):
        run_map_training(
            config, {"train": loader, "cifar10_test": loader}
        )


def test_failed_checkpoint_write_preserves_previous_state(
    tmp_path, monkeypatch
):
    import code.engine as engine

    path = tmp_path / "state.pt"
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_training_state(path, model, optimizer, epoch=0, history=[])
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        model.weight.add_(1.0)

    def fail_after_partial_write(state, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(engine.torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_training_state(path, model, optimizer, epoch=1, history=[])

    restored = nn.Linear(2, 2)
    load_training_state(path, restored)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])
    assert not (tmp_path / "state.pt.tmp").exists()
