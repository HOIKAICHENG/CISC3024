import copy
import hashlib
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
    map_checkpoint_sha256: str,
) -> dict:
    return {
        "stage": "abnn",
        "quick": quick,
        "epochs": epochs,
        "member_index": index,
        "member_seed": seed,
        "map_checkpoint_sha256": map_checkpoint_sha256,
        "config": config.to_dict(),
    }


def _checkpoint_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_map_metadata(metadata, quick: bool) -> None:
    required = {"stage", "quick"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        missing = required.difference(metadata if isinstance(metadata, dict) else ())
        raise ValueError(
            "MAP checkpoint is missing required metadata fields: "
            f"{sorted(missing)}"
        )
    if metadata["stage"] != "map":
        raise ValueError(
            "ABNN adaptation requires a MAP checkpoint, "
            f"got stage {metadata['stage']!r}"
        )
    if metadata["quick"] is not quick:
        expected = "quick" if quick else "full"
        actual = "quick" if metadata["quick"] is True else "full"
        raise ValueError(
            "MAP checkpoint quick/full mode mismatch: "
            f"expected {expected}, got {actual}"
        )


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
    _validate_map_metadata(source_metadata, quick)
    map_checkpoint_sha256 = _checkpoint_sha256(map_checkpoint)
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
        metadata = _member_metadata(
            config,
            epochs,
            quick,
            index,
            seed,
            map_checkpoint_sha256,
        )
        history = []
        start = 0
        if checkpoint.exists():
            try:
                state = load_training_state(
                    checkpoint,
                    model,
                    optimizer,
                    restore_rng=True,
                    expected_metadata=metadata,
                )
            except ValueError as error:
                if "metadata is incompatible" not in str(error):
                    raise
                raise ValueError(
                    f"ABNN member {index} checkpoint is incompatible with "
                    "the originating MAP checkpoint or current run settings"
                ) from error
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
