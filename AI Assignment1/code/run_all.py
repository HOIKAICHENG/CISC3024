"""Reproduce every experiment in the report.

    python run_all.py            # full pipeline (about 1.5 h on a 20-core CPU)
    python run_all.py --quick    # smoke-test the pipeline in a few minutes

Stage 1  FCMAE pre-training, ConvNeXt V2 (GRN) and V1 (no GRN)                [parallel]
Stage 2  supervised training on 5k labels: {V1, V2} x {scratch, FCMAE init}  [parallel]
Stage 3  analysis (feature collapse, k-NN, reconstructions) + figures
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def run_parallel(cmds: list[list[str]], threads: int):
    procs = []
    for cmd in cmds:
        full = [sys.executable, *cmd, "--threads", str(threads)]
        print("launch:", " ".join(full), flush=True)
        procs.append(subprocess.Popen(full, cwd=HERE))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a stage failed with exit codes {codes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--skip-pretrain", action="store_true", help="reuse existing checkpoints")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(os.path.dirname(HERE), "data", "cifar-10-batches-py")):
        subprocess.check_call([sys.executable, os.path.join(HERE, "prepare_data.py")])

    pre_extra = ["--limit-batches", "5"] if args.quick else []
    ft_extra = ["--n-per-class", "20", "--eval-every", "1"] if args.quick else []
    pre_epochs = 1 if args.quick else args.pretrain_epochs
    ft_epochs = 1 if args.quick else args.finetune_epochs

    if not args.skip_pretrain:
        run_parallel(
            [
                ["pretrain.py", "--epochs", str(pre_epochs), "--tag", "v2_grn", *pre_extra],
                ["pretrain.py", "--epochs", str(pre_epochs), "--tag", "v1_nogrn", "--no-grn", *pre_extra],
            ],
            threads=10,
        )

    v2 = os.path.join(RESULTS, "fcmae_v2_grn.pth")
    v1 = os.path.join(RESULTS, "fcmae_v1_nogrn.pth")
    run_parallel(
        [
            ["finetune.py", "--epochs", str(ft_epochs), "--tag", "v2_fcmae", "--init", v2, *ft_extra],
            ["finetune.py", "--epochs", str(ft_epochs), "--tag", "v2_scratch", *ft_extra],
            ["finetune.py", "--epochs", str(ft_epochs), "--tag", "v1_fcmae", "--init", v1, "--no-grn", *ft_extra],
            ["finetune.py", "--epochs", str(ft_epochs), "--tag", "v1_scratch", "--no-grn", *ft_extra],
        ],
        threads=4,
    )

    knn = ["--knn-train", "2000"] if args.quick else []
    subprocess.check_call([sys.executable, os.path.join(HERE, "analysis.py"), *knn], cwd=HERE)
    subprocess.check_call([sys.executable, os.path.join(HERE, "make_figures.py")], cwd=HERE)
    print("all done; see ../results and ../figures")


if __name__ == "__main__":
    main()
