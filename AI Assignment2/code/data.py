import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import ExperimentConfig, QUICK_DATASET_LIMITS

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def deterministic_subset_indices(size: int, count: int, seed: int) -> list[int]:
    if not 0 < count <= size:
        raise ValueError(f"subset count {count} must be in [1, {size}]")
    return random.Random(seed).sample(range(size), count)


def build_loaders(
    config: ExperimentConfig, quick: bool = False
) -> dict[str, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    root = config.data_dir
    sets = {
        "train": datasets.CIFAR10(
            root, train=True, download=True, transform=train_tf
        ),
        "cifar10_test": datasets.CIFAR10(
            root, train=False, download=True, transform=test_tf
        ),
        "svhn_test": datasets.SVHN(
            root, split="test", download=True, transform=test_tf
        ),
        "cifar100_test": datasets.CIFAR100(
            root, train=False, download=True, transform=test_tf
        ),
    }
    if quick:
        sets = {
            name: Subset(
                ds,
                deterministic_subset_indices(
                    len(ds),
                    QUICK_DATASET_LIMITS[name],
                    config.seed,
                ),
            )
            for name, ds in sets.items()
        }
    return {
        name: DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=config.num_workers,
            pin_memory=False,
        )
        for name, ds in sets.items()
    }
