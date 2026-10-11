from dataclasses import asdict, dataclass
import random

import numpy as np
import torch

DATASET_NAMES = ("train", "cifar10_test", "svhn_test", "cifar100_test")
QUICK_DATASET_LIMITS = {
    "train": 1000,
    "cifar10_test": 500,
    "svhn_test": 500,
    "cifar100_test": 500,
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    member_seeds: tuple[int, ...] = (101, 102, 103)
    batch_size: int = 128
    map_epochs: int = 20
    abnn_epochs: int = 5
    map_lr: float = 0.1
    abnn_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    mc_samples: int = 3
    emphasized_classes: int = 3
    emphasis_factor: float = 5.0
    noise_std: float = 0.1
    num_workers: int = 2
    data_dir: str = "data"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


def run_metadata(config: ExperimentConfig, quick: bool) -> dict:
    if not isinstance(quick, bool):
        raise TypeError("quick must be a bool")
    return {
        "mode": "quick" if quick else "full",
        "quick": quick,
        "effective": {
            "dataset_limits": (
                dict(QUICK_DATASET_LIMITS)
                if quick
                else {name: None for name in DATASET_NAMES}
            ),
            "map_epochs": 1 if quick else config.map_epochs,
            "abnn_epochs": 1 if quick else config.abnn_epochs,
            "member_seeds": list(
                config.member_seeds[:1] if quick else config.member_seeds
            ),
            "member_count": 1 if quick else len(config.member_seeds),
            "mc_samples": 2 if quick else 3,
        },
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
