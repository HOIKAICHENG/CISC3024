from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from .abnn import monte_carlo_probabilities
from .data import CIFAR_MEAN, CIFAR_STD
from .engine import deterministic_probabilities, save_json
from .metrics import classification_metrics, ood_metrics, predictive_entropy

OOD_DATASETS = ("svhn", "cifar100")


def _validate_loader_order(loaders) -> None:
    for name in ("cifar10", *OOD_DATASETS):
        loader = loaders[f"{name}_test"]
        if type(getattr(loader, "sampler", None)) is not SequentialSampler:
            raise ValueError(
                f"{name}_test must use SequentialSampler for "
                "deterministic dataset order"
            )


def _validate_target_order(reference, actual, dataset, method) -> None:
    if not np.array_equal(reference, actual):
        raise RuntimeError(
            f"{dataset} target order differs between MAP and {method}"
        )


def _method_result(id_probabilities, targets, ood_probabilities):
    id_entropy = predictive_entropy(id_probabilities)
    ood_entropy = {
        name: predictive_entropy(probabilities)
        for name, probabilities in ood_probabilities.items()
    }
    return {
        "classification": classification_metrics(id_probabilities, targets),
        "ood": {
            name: ood_metrics(id_entropy, entropy)
            for name, entropy in ood_entropy.items()
        },
        "ood_predictive_entropy": {
            name: entropy.tolist() for name, entropy in ood_entropy.items()
        },
        "probabilities": id_probabilities.tolist(),
        "targets": targets.tolist(),
        "predictive_entropy": id_entropy.tolist(),
    }


def _example_images(loader, count: int = 16):
    images, targets = next(iter(loader))
    mean = images.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    images = (images[:count] * std + mean).clamp(0, 1)
    return images.tolist(), targets[:count].tolist()


def evaluate_methods(
    config, loaders, map_model, members, quick: bool = False
) -> dict:
    expected_members = 1 if quick else 3
    mode = "quick" if quick else "full"
    if len(members) != expected_members:
        raise ValueError(
            f"{mode} evaluation requires exactly {expected_members} "
            f"ABNN member{'s' if expected_members != 1 else ''}"
        )
    _validate_loader_order(loaders)
    device = torch.device("cpu")
    samples = 2 if quick else 3
    map_id, targets = deterministic_probabilities(
        map_model, loaders["cifar10_test"], device
    )
    map_ood = {}
    reference_targets = {"cifar10": targets}
    for name in OOD_DATASETS:
        probabilities, dataset_targets = deterministic_probabilities(
            map_model, loaders[f"{name}_test"], device
        )
        map_ood[name] = probabilities
        reference_targets[name] = dataset_targets
    methods = {"map": _method_result(map_id, targets, map_ood)}

    for name, selected in (
        ("abnn_single", members[:1]),
        ("abnn_full", members),
    ):
        id_probabilities, member_targets, _ = monte_carlo_probabilities(
            selected, loaders["cifar10_test"], samples, device
        )
        _validate_target_order(
            targets, member_targets, "cifar10", name
        )
        ood_probabilities = {}
        for dataset in OOD_DATASETS:
            probabilities, dataset_targets, _ = monte_carlo_probabilities(
                selected, loaders[f"{dataset}_test"], samples, device
            )
            _validate_target_order(
                reference_targets[dataset],
                dataset_targets,
                dataset,
                name,
            )
            ood_probabilities[dataset] = probabilities
        methods[name] = _method_result(
            id_probabilities, targets, ood_probabilities
        )

    images, image_targets = _example_images(loaders["cifar10_test"])
    if not np.array_equal(
        targets[: len(image_targets)], np.asarray(image_targets)
    ):
        raise RuntimeError(
            "example target order differs from saved CIFAR-10 predictions"
        )
    payload = {
        "config": config.to_dict(),
        "methods": methods,
        "example_images": images,
        "example_targets": image_targets,
    }
    suffix = "_quick" if quick else ""
    save_json(
        Path(config.results_dir) / f"evaluation{suffix}.json", payload
    )
    return payload
