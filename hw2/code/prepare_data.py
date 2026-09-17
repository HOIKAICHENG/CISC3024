"""Fetch CIFAR-10 and write it in the layout expected by torchvision.

The canonical cs.toronto.edu tarball downloads at ~70 kB/s from our network, so
the identical data is pulled from the HuggingFace mirror (uoft-cs/cifar10) and
repacked into `data/cifar-10-batches-py/` so that
`torchvision.datasets.CIFAR10(..., download=False)` works unchanged.
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import urllib.request

import numpy as np

BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(ROOT, "cifar-10-batches-py")
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached {os.path.basename(dest)}")
        return
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got/1e6:6.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()


def parquet_to_arrays(path: str):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path)
    img_col = "img" if "img" in table.column_names else "image"
    images = table.column(img_col).to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)

    data = np.zeros((len(images), 3072), dtype=np.uint8)
    for i, rec in enumerate(images):
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        data[i] = arr.transpose(2, 0, 1).reshape(-1)  # CIFAR stores channel-major
    return data, labels


def write_batch(path: str, data: np.ndarray, labels: np.ndarray) -> None:
    # torchvision loads with encoding="latin1" and indexes str keys
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels.tolist()}, f)


def main() -> int:
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    for split, name in FILES.items():
        download(BASE + name, os.path.join(ROOT, name))

    print("decoding train split ...")
    tr_data, tr_labels = parquet_to_arrays(os.path.join(ROOT, FILES["train"]))
    print("decoding test split ...")
    te_data, te_labels = parquet_to_arrays(os.path.join(ROOT, FILES["test"]))
    assert tr_data.shape == (50000, 3072) and te_data.shape == (10000, 3072)

    for i in range(5):
        sl = slice(i * 10000, (i + 1) * 10000)
        write_batch(os.path.join(OUT, f"data_batch_{i+1}"), tr_data[sl], tr_labels[sl])
    write_batch(os.path.join(OUT, "test_batch"), te_data, te_labels)
    with open(os.path.join(OUT, "batches.meta"), "wb") as f:
        pickle.dump({"label_names": CLASSES}, f)

    from data import CIFAR10Local

    tr = CIFAR10Local(ROOT, train=True, download=False)
    te = CIFAR10Local(ROOT, train=False, download=False)
    print(f"OK: train={len(tr)} test={len(te)} example={tr[0][0].size} label={CLASSES[tr[0][1]]}")
    print("class counts (train):", np.bincount(np.asarray(tr.targets)).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
