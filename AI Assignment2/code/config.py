from dataclasses import asdict, dataclass
import random

import numpy as np
import torch


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
    num_workers: int = 2
    data_dir: str = "data"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
