"""CIFAR-10 data pipeline for the FCMAE / ConvNeXt V2 experiments."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class CIFAR10Local(datasets.CIFAR10):
    """CIFAR-10 read from locally repacked batches (see `prepare_data.py`).

    The pickles are rebuilt from the HuggingFace mirror, so their MD5s differ
    from the ones torchvision hard-codes for the cs.toronto.edu tarball; the
    checksums are disabled and only file presence is verified.
    """

    train_list = [[f"data_batch_{i}", None] for i in range(1, 6)]
    test_list = [["test_batch", None]]
    meta = {"filename": "batches.meta", "key": "label_names", "md5": None}

    def download(self) -> None:  # pragma: no cover - data is prepared offline
        raise RuntimeError("run `python prepare_data.py` first")


def pretrain_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.6, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def stratified_subset_indices(targets, n_per_class: int, seed: int = 0):
    """Pick `n_per_class` examples of every class (deterministic given `seed`)."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    idx = []
    for c in np.unique(targets):
        c_idx = np.where(targets == c)[0]
        idx.extend(rng.choice(c_idx, n_per_class, replace=False).tolist())
    return sorted(idx)


def build_loaders(
    batch_size: int = 128,
    n_per_class: int | None = 500,
    mode: str = "finetune",
    num_workers: int = 4,
    seed: int = 0,
):
    """`mode='pretrain'` returns the unlabeled 50k train split, otherwise a
    labelled subset plus the full 10k test split."""
    if mode == "pretrain":
        ds = CIFAR10Local(DATA_ROOT, train=True, download=False, transform=pretrain_transform())
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        return loader, None

    train_full = CIFAR10Local(DATA_ROOT, train=True, download=False, transform=train_transform())
    if n_per_class is not None:
        idx = stratified_subset_indices(train_full.targets, n_per_class, seed=seed)
        train_ds = Subset(train_full, idx)
    else:
        train_ds = train_full
    test_ds = CIFAR10Local(DATA_ROOT, train=False, download=False, transform=eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
