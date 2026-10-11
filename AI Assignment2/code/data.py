"""CIFAR-10 (in-distribution) and SVHN (OOD) loaders."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

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


class CIFAR10Local(datasets.CIFAR10):
    """CIFAR-10 from locally repacked batches (see ``prepare_data.py``)."""

    train_list = [[f"data_batch_{i}", None] for i in range(1, 6)]
    test_list = [["test_batch", None]]
    meta = {"filename": "batches.meta", "key": "label_names", "md5": None}

    def download(self) -> None:  # pragma: no cover
        raise RuntimeError("run `python code/prepare_data.py` first")


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


def _loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )


def build_cifar_loaders(batch_size: int = 128, num_workers: int = 0):
    train_ds = CIFAR10Local(DATA_ROOT, train=True, download=False, transform=train_transform())
    test_ds = CIFAR10Local(DATA_ROOT, train=False, download=False, transform=eval_transform())
    train = _loader(train_ds, batch_size, True, num_workers, drop_last=True)
    test = _loader(test_ds, 256, False, num_workers)
    return train, test


def build_svhn_loader(batch_size: int = 256, num_workers: int = 0, max_samples: int | None = 10000):
    from prepare_data import SVHNLocal

    ds = SVHNLocal(DATA_ROOT, transform=eval_transform(), max_samples=max_samples)
    return _loader(ds, batch_size, False, num_workers)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
