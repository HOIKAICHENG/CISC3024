import json
from pathlib import Path
import random

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
    return total_loss / total


@torch.inference_mode()
def deterministic_probabilities(model, loader, device):
    model.eval()
    probabilities = []
    targets = []
    for inputs, labels in loader:
        probabilities.append(model(inputs.to(device)).softmax(dim=1).cpu())
        targets.append(labels.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                default=_json_default,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_state(
    path,
    model,
    optimizer,
    epoch: int,
    history: list[dict],
    *,
    metadata: dict | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        },
    }
    if metadata is not None:
        state["metadata"] = metadata
    if elapsed_seconds is not None:
        state["elapsed_seconds"] = elapsed_seconds
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_rng_state(rng_state) -> None:
    required = {"python", "numpy", "torch"}
    if not isinstance(rng_state, dict) or set(rng_state) != required:
        raise ValueError(
            "checkpoint RNG state must contain python, numpy, and torch states"
        )
    try:
        random.Random().setstate(rng_state["python"])
        np.random.RandomState().set_state(rng_state["numpy"])
        torch.Generator(device="cpu").set_state(rng_state["torch"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"checkpoint RNG state is invalid: {error}") from error


def load_training_state(
    path,
    model,
    optimizer=None,
    *,
    restore_rng: bool = False,
    expected_metadata: dict | None = None,
) -> dict:
    path = Path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(
            f"could not load checkpoint '{path}': {error}"
        ) from error
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a dictionary")
    required = {
        "format_version",
        "model",
        "optimizer",
        "epoch",
        "history",
        "rng_state",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            f"checkpoint is missing required fields: {sorted(missing)}"
        )
    if (
        type(state["format_version"]) is not int
        or state["format_version"] != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "checkpoint format_version "
            f"{state['format_version']!r} is unsupported; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(state["model"], dict):
        raise ValueError("checkpoint model state must be a dictionary")
    if not isinstance(state["optimizer"], dict):
        raise ValueError("checkpoint optimizer state must be a dictionary")
    if (
        isinstance(state["epoch"], bool)
        or not isinstance(state["epoch"], int)
        or state["epoch"] < 0
    ):
        raise ValueError("checkpoint epoch must be a non-negative integer")
    if not isinstance(state["history"], list):
        raise ValueError("checkpoint history must be a list")
    _validate_rng_state(state["rng_state"])
    if (
        expected_metadata is not None
        and state.get("metadata") != expected_metadata
    ):
        raise ValueError(
            "checkpoint metadata is incompatible with this training run"
        )
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        rng_state = state["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch"])
    return state
