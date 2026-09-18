"""Turn the JSON logs in ../results into the figures used by the report."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data import CLASSES
from engine import RESULTS_DIR

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
FT_RUNS = {
    "v2_fcmae": "ConvNeXt V2 (GRN) + FCMAE pre-training",
    "v2_scratch": "ConvNeXt V2 (GRN), from scratch",
    "v1_fcmae": "ConvNeXt V1 (no GRN) + FCMAE pre-training",
    "v1_scratch": "ConvNeXt V1 (no GRN), from scratch",
}
COLORS = {"v2_fcmae": "#d62728", "v2_scratch": "#ff9896", "v1_fcmae": "#1f77b4", "v1_scratch": "#aec7e8"}


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_pretrain_loss():
    runs = {"ConvNeXt V2 (GRN)": load("pretrain_v2_grn.json"),
            "ConvNeXt V1 (no GRN)": load("pretrain_v1_nogrn.json")}
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for (name, r), c in zip(runs.items(), ("#d62728", "#1f77b4")):
        if r is None:
            continue
        ep = [h["epoch"] for h in r["history"]]
        ax.plot(ep, [h["loss"] for h in r["history"]], marker="o", color=c, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("masked-patch MSE (normalised pixels)")
    ax.set_title("FCMAE pre-training loss on CIFAR-10 (50k unlabeled images)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_pretrain_loss.png"), dpi=200)
    plt.close(fig)


def fig_finetune_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for tag, label in FT_RUNS.items():
        r = load(f"finetune_{tag}.json")
        if r is None:
            continue
        ls = "-" if "fcmae" in tag else "--"
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], ls, color=COLORS[tag], label=label)
        pts = [(h["epoch"], h["test_acc"]) for h in r["history"] if h["test_acc"] is not None]
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", ms=3,
                     color=COLORS[tag], label=label)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss (CE, label smoothing 0.1)")
    axes[0].set_title("Training loss (5,000 labelled images)", fontsize=9)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy (%)")
    axes[1].set_title("Top-1 accuracy on the 10,000-image test set", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_finetune_curves.png"), dpi=200)
    plt.close(fig)


def fig_accuracy_bars():
    an = load("analysis.json") or {}
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    groups = ["ConvNeXt V1\n(no GRN)", "ConvNeXt V2\n(GRN)"]
    x = np.arange(len(groups))
    w = 0.25
    series = {
        "k-NN on frozen FCMAE features": [
            (an.get("knn_acc") or {}).get("ConvNeXt V1 (no GRN)"),
            (an.get("knn_acc") or {}).get("ConvNeXt V2 (GRN)"),
        ],
        "supervised from scratch": [
            (load("finetune_v1_scratch.json") or {}).get("final_acc"),
            (load("finetune_v2_scratch.json") or {}).get("final_acc"),
        ],
        "FCMAE pre-train + fine-tune": [
            (load("finetune_v1_fcmae.json") or {}).get("final_acc"),
            (load("finetune_v2_fcmae.json") or {}).get("final_acc"),
        ],
    }
    for i, (name, vals) in enumerate(series.items()):
        vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Effect of GRN and FCMAE pre-training (5k labels for fine-tuning)", fontsize=9)
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_accuracy_bars.png"), dpi=200)
    plt.close(fig)


def fig_confusion():
    r = load("finetune_v2_fcmae.json")
    if r is None:
        return
    cm = np.asarray(r["confusion"], dtype=float)
    cm_n = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm_n[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if cm_n[i, j] > 50 else "black")
    ax.set_title(f"Confusion matrix (%), ConvNeXt V2 + FCMAE, acc = {r['final_acc']:.2f}%", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_confusion.png"), dpi=200)
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc="#f0f0f0", ec="#333333", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 color="#333333"))


def fig_architecture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.05, 1.5]})
    # ---- (a) ConvNeXt V2 block
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
    ax1.set_title("(a) ConvNeXt V2 block", fontsize=9, fontweight="bold")
    steps = [
        ("input  (N, C, H, W)", "#ffffff"),
        ("7x7 depthwise conv, C", "#dbe9f6"),
        ("LayerNorm", "#e8e8e8"),
        ("1x1 conv  C -> 4C", "#dbe9f6"),
        ("GELU", "#e8e8e8"),
        ("GRN  (new in V2, replaces LayerScale)", "#f9d4d4"),
        ("1x1 conv  4C -> C", "#dbe9f6"),
        ("+ residual  (drop-path)", "#ffffff"),
    ]
    y = 9.3
    for i, (txt, fc) in enumerate(steps):
        _box(ax1, 1.6, y - 0.75, 6.8, 0.75, txt, fc=fc, bold=(i == 5))
        if i < len(steps) - 1:
            _arrow(ax1, 5.0, y - 0.78, 5.0, y - 1.15)
        y -= 1.17
    y_last = 9.3 - (len(steps) - 1) * 1.17 - 0.375
    ax1.plot([1.6, 0.8, 0.8, 1.6], [9.3 - 0.375, 9.3 - 0.375, y_last, y_last],
             color="#333333", lw=1.0)
    ax1.text(0.35, 5.2, "skip", rotation=90, fontsize=7, va="center")

    # ---- (b) FCMAE
    ax2.set_xlim(0, 15); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("(b) FCMAE pre-training (fully convolutional masked autoencoder)", fontsize=9,
                  fontweight="bold")
    _box(ax2, 0.3, 6.0, 2.6, 2.4, "image\n32x32x3", fc="#ffffff")
    _arrow(ax2, 2.9, 7.2, 3.6, 7.2)
    _box(ax2, 3.6, 6.0, 3.0, 2.4, "random mask\n60% of 4x4\npatches removed", fc="#fff2cc")
    _arrow(ax2, 6.6, 7.2, 7.3, 7.2)
    _box(ax2, 7.3, 5.2, 3.4, 4.0, "ConvNeXt V2\nencoder\n(4 stages)\nmask re-applied\nafter every stage",
         fc="#dbe9f6", bold=True)
    _arrow(ax2, 10.7, 7.2, 11.4, 7.2)
    _box(ax2, 11.4, 6.0, 3.3, 2.4, "1x1 proj + upsample\n+ mask tokens", fc="#e8e8e8")
    _arrow(ax2, 13.05, 6.0, 13.05, 5.3)
    _box(ax2, 11.4, 3.0, 3.3, 2.3, "decoder:\n1 ConvNeXt V2 block\n+ 1x1 pred head", fc="#f9d4d4")
    _arrow(ax2, 11.4, 4.15, 10.7, 4.15)
    _box(ax2, 7.3, 3.0, 3.4, 2.3, "reconstruct\nnormalised pixels\nof masked patches", fc="#ffffff")
    _arrow(ax2, 7.3, 4.15, 6.6, 4.15)
    _box(ax2, 3.6, 3.0, 3.0, 2.3, "MSE loss\n(masked patches only)", fc="#fff2cc")
    ax2.text(7.5, 1.6, "After pre-training: keep the encoder, add global-average-pool + linear head,\n"
                       "fine-tune on labelled images.", fontsize=7.5, ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_architecture.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_architecture()
    fig_pretrain_loss()
    fig_finetune_curves()
    fig_accuracy_bars()
    fig_confusion()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
