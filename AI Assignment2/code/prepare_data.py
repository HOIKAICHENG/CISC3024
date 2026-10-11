"""Fetch CIFAR-10 and the SVHN test set into ``data/``.

CIFAR-10 is pulled from the HuggingFace ``uoft-cs/cifar10`` mirror (the official
Toronto tarball is too slow from this network) and repacked for torchvision.
SVHN's test split is the official ``test_32x32.mat`` (32x32 house-number
images), used as the out-of-distribution set exactly as in the ABNN paper.
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import urllib.request

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CIFAR_OUT = os.path.join(ROOT, "cifar-10-batches-py")
CIFAR_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
CIFAR_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
SVHN_PARQUET = (
    "https://huggingface.co/datasets/ufldl-stanford/svhn/resolve/"
    "refs%2Fconvert%2Fparquet/cropped_digits/test/0000.parquet"
)
SVHN_MAT_URLS = [
    "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
]
CLASSES = [
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
]


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached {os.path.basename(dest)}")
        return
    print(f"  downloading {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got / 1e6:6.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()


def download_first(urls: list[str], dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached {os.path.basename(dest)}")
        return
    last = None
    for url in urls:
        try:
            download(url, dest)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  failed {url}: {exc}")
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(f"could not download SVHN: {last}")


def parquet_to_arrays(path: str):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    img_col = "img" if "img" in table.column_names else "image"
    images = table.column(img_col).to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    data = np.zeros((len(images), 3072), dtype=np.uint8)
    for i, rec in enumerate(images):
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        data[i] = arr.transpose(2, 0, 1).reshape(-1)
    return data, labels


def write_batch(path: str, data: np.ndarray, labels: np.ndarray) -> None:
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels.tolist()}, f)


class SVHNLocal(Dataset):
    """SVHN test images stored as a cached ``.npz`` next to the official ``.mat``."""

    def __init__(self, root: str, transform=None, max_samples: int | None = None):
        cache = os.path.join(root, "svhn_test.npz")
        if not os.path.exists(cache):
            raise RuntimeError("run `python code/prepare_data.py` first")
        blob = np.load(cache)
        self.data = blob["data"]
        self.labels = blob["labels"]
        if max_samples is not None:
            self.data = self.data[:max_samples]
            self.labels = self.labels[:max_samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def convert_svhn_mat(mat_path: str, cache_path: str) -> None:
    from scipy.io import loadmat

    mat = loadmat(mat_path)
    images = np.transpose(mat["X"], (3, 0, 1, 2)).astype(np.uint8)
    labels = mat["y"].reshape(-1).astype(np.int64)
    labels = np.where(labels == 10, 0, labels)
    np.savez_compressed(cache_path, data=images, labels=labels)
    print(f"  SVHN test: {len(labels)} images -> {cache_path}")


def convert_svhn_parquet(parquet_path: str, cache_path: str) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    img_col = "image" if "image" in table.column_names else "img"
    lab_col = "label" if "label" in table.column_names else "labels"
    images = table.column(img_col).to_pylist()
    labels = np.asarray(table.column(lab_col).to_pylist(), dtype=np.int64)
    labels = np.where(labels == 10, 0, labels)
    data = np.zeros((len(images), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(images):
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        data[i] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    np.savez_compressed(cache_path, data=data, labels=labels)
    print(f"  SVHN test: {len(labels)} images -> {cache_path}")


def prepare_cifar10() -> None:
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(CIFAR_OUT, exist_ok=True)
    marker = os.path.join(CIFAR_OUT, "test_batch")
    if os.path.exists(marker) and os.path.getsize(marker) > 0:
        print("CIFAR-10 already prepared")
        return

    for _split, name in CIFAR_FILES.items():
        download(CIFAR_BASE + name, os.path.join(ROOT, name))

    print("decoding CIFAR-10 train ...")
    tr_data, tr_labels = parquet_to_arrays(os.path.join(ROOT, CIFAR_FILES["train"]))
    print("decoding CIFAR-10 test ...")
    te_data, te_labels = parquet_to_arrays(os.path.join(ROOT, CIFAR_FILES["test"]))
    assert tr_data.shape == (50000, 3072) and te_data.shape == (10000, 3072)

    for i in range(5):
        sl = slice(i * 10000, (i + 1) * 10000)
        write_batch(os.path.join(CIFAR_OUT, f"data_batch_{i + 1}"), tr_data[sl], tr_labels[sl])
    write_batch(os.path.join(CIFAR_OUT, "test_batch"), te_data, te_labels)
    with open(os.path.join(CIFAR_OUT, "batches.meta"), "wb") as f:
        pickle.dump({"label_names": CLASSES}, f)
    print(f"CIFAR-10 ready: train={len(tr_labels)} test={len(te_labels)}")


def prepare_svhn() -> None:
    os.makedirs(ROOT, exist_ok=True)
    cache = os.path.join(ROOT, "svhn_test.npz")
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        print("SVHN already prepared")
        return
    parquet_path = os.path.join(ROOT, "svhn_test.parquet")
    try:
        download(SVHN_PARQUET, parquet_path)
        convert_svhn_parquet(parquet_path, cache)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"  HF parquet failed: {exc}")
    mat_path = os.path.join(ROOT, "test_32x32.mat")
    download_first(SVHN_MAT_URLS, mat_path)
    convert_svhn_mat(mat_path, cache)


def main() -> int:
    prepare_cifar10()
    prepare_svhn()
    from data import CIFAR10Local

    tr = CIFAR10Local(ROOT, train=True, download=False)
    te = CIFAR10Local(ROOT, train=False, download=False)
    ood = SVHNLocal(ROOT)
    print(f"OK: cifar-train={len(tr)} cifar-test={len(te)} svhn-test={len(ood)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
