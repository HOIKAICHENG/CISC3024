import math
from pathlib import Path
import time

import torch
from torch import nn

from .config import ExperimentConfig, seed_everything
from .engine import (
    deterministic_probabilities,
    load_training_state,
    save_json,
    save_training_state,
    train_epoch,
)
from .metrics import classification_metrics
from .model import resnet20


def _lr_factor(epoch: int, epochs: int) -> float:
    if epoch == 0:
        return 0.1
    progress = (epoch - 1) / max(1, epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _checkpoint_metadata(
    config: ExperimentConfig, epochs: int, quick: bool
) -> dict:
    return {
        "stage": "map",
        "quick": quick,
        "epochs": epochs,
        "config": config.to_dict(),
    }


def _validate_resume_state(state: dict, epochs: int) -> None:
    start = state["epoch"]
    history = state["history"]
    if start > epochs:
        raise ValueError(
            f"checkpoint epoch {start} exceeds target epoch count {epochs}"
        )
    if len(history) != start:
        raise ValueError(
            "checkpoint history length does not match its completed epoch"
        )
    if any(
        not isinstance(row, dict) or row.get("epoch") != index
        for index, row in enumerate(history, start=1)
    ):
        raise ValueError("checkpoint history contains inconsistent epoch rows")


def run_map_training(
    config: ExperimentConfig, loaders, quick: bool = False
):
    seed_everything(config.seed)
    device = torch.device("cpu")
    model = resnet20().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.map_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    epochs = 1 if quick else config.map_epochs
    suffix = "_quick" if quick else ""
    checkpoint = Path(config.results_dir) / f"map_checkpoint{suffix}.pt"
    history_path = Path(config.results_dir) / f"map_history{suffix}.json"
    metadata = _checkpoint_metadata(config, epochs, quick)

    history = []
    start = 0
    previous_elapsed = 0.0
    if checkpoint.exists():
        state = load_training_state(
            checkpoint,
            model,
            optimizer,
            restore_rng=True,
            expected_metadata=metadata,
        )
        _validate_resume_state(state, epochs)
        history = state["history"]
        start = state["epoch"]
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))

    elapsed_seconds = previous_elapsed
    started = time.perf_counter()
    for epoch in range(start, epochs):
        learning_rate = config.map_lr * _lr_factor(epoch, epochs)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            nn.CrossEntropyLoss(),
            device,
        )
        probabilities, targets = deterministic_probabilities(
            model, loaders["cifar10_test"], device
        )
        row = {
            "epoch": epoch + 1,
            "loss": loss,
            "lr": learning_rate,
            **classification_metrics(probabilities, targets),
        }
        history.append(row)
        elapsed_seconds = previous_elapsed + time.perf_counter() - started
        save_training_state(
            checkpoint,
            model,
            optimizer,
            epoch=epoch + 1,
            history=history,
            metadata=metadata,
            elapsed_seconds=elapsed_seconds,
        )
        save_json(
            history_path,
            {
                "config": config.to_dict(),
                "history": history,
                "elapsed_seconds": elapsed_seconds,
            },
        )
    save_json(
        history_path,
        {
            "config": config.to_dict(),
            "history": history,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    return model, history
