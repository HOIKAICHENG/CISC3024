import copy
from pathlib import Path

import torch
from torch import nn

from .abnn import (
    class_weights_for_member,
    convert_to_abnn,
    freeze_except_bayesian_norm,
)
from .config import ExperimentConfig, seed_everything
from .engine import (
    load_training_state,
    save_json,
    save_training_state,
    train_epoch,
)
from .model import resnet20


def _member_metadata(
    config: ExperimentConfig,
    epochs: int,
    quick: bool,
    index: int,
    seed: int,
) -> dict:
    return {
        "stage": "abnn",
        "quick": quick,
        "epochs": epochs,
        "member_index": index,
        "member_seed": seed,
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


def run_abnn_training(
    config: ExperimentConfig, loaders, map_checkpoint, quick: bool = False
):
    if "train" not in loaders:
        raise ValueError(
            "ABNN adaptation requires the in-distribution training loader"
        )
    base = resnet20()
    source = load_training_state(map_checkpoint, base)
    source_metadata = source.get("metadata")
    source_stage = (
        source_metadata.get("stage")
        if isinstance(source_metadata, dict)
        else None
    )
    if source_stage not in (None, "map"):
        raise ValueError(
            "ABNN adaptation requires a MAP checkpoint, "
            f"got stage {source_stage!r}"
        )
    seeds = config.member_seeds[:1] if quick else config.member_seeds
    epochs = 1 if quick else config.abnn_epochs
    suffix = "_quick" if quick else ""
    device = torch.device("cpu")
    members, histories = [], []
    for index, seed in enumerate(seeds):
        seed_everything(seed)
        model = convert_to_abnn(copy.deepcopy(base))
        freeze_except_bayesian_norm(model)
        optimizer = torch.optim.SGD(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=config.abnn_lr,
            momentum=config.momentum,
            weight_decay=0.0,
        )
        weights, selected = class_weights_for_member(
            10,
            config.emphasized_classes,
            config.emphasis_factor,
            seed,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        checkpoint = Path(config.results_dir) / f"abnn_member_{index}{suffix}.pt"
        metadata = _member_metadata(config, epochs, quick, index, seed)
        history = []
        start = 0
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
        for epoch in range(start, epochs):
            loss = train_epoch(
                model, loaders["train"], optimizer, criterion, device
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss,
                    "emphasized_classes": selected,
                }
            )
            save_training_state(
                checkpoint,
                model,
                optimizer,
                epoch=epoch + 1,
                history=history,
                metadata=metadata,
            )
        members.append(model)
        histories.append(list(history))
    save_json(
        Path(config.results_dir) / f"abnn_history{suffix}.json",
        {"config": config.to_dict(), "members": histories},
    )
    return members, histories
