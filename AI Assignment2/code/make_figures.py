"""All report figures, generated from ``results/*.json``."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def J(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup():
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def box(ax, xy, w, h, text, fc, ec="#222"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8.5)
    return p


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    box(ax, (0.2, 1.1), 2.4, 1.2, "1. Train MAP DNN\nResNet-20 + BatchNorm\nCIFAR-10, 20 epochs", "#D6EAF8")
    box(ax, (3.3, 1.1), 2.6, 1.2, "2. Replace BN → BNL\nγ ← γ(1+ε), ε~N(0,1)\ncopy running stats", "#FDEBD0")
    box(ax, (6.6, 1.1), 3.3, 1.2, "3. Fine-tune M ABNNs\nonly γ, β  ·  class-weighted CE\nthen average M×L samples", "#D5F5E3")
    for x in (2.7, 6.0):
        ax.annotate("", xy=(x + 0.5, 1.7), xytext=(x, 1.7),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    ax.set_title("ABNN training pipeline (Franchi et al., CVPR 2024)")
    fig.savefig(os.path.join(FIG, "fig_pipeline.png"))
    plt.close()


def fig_bnl():
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    steps = [
        (0.2, "h", "pre-activation", "#E8DAEF"),
        (2.0, "normalize", r"$(h-\mu)/\sigma$", "#D6EAF8"),
        (4.2, "noisy scale", r"$\gamma(1+\epsilon)$", "#F5B7B1"),
        (6.4, "shift", r"$+\ \beta$", "#D5F5E3"),
        (8.1, "u", "BNL output", "#FDEBD0"),
    ]
    for x, title, sub, fc in steps:
        box(ax, (x, 1.15), 1.4, 1.15, f"{title}\n{sub}", fc)
    for x in (1.65, 3.45, 5.65, 7.85):
        ax.annotate("", xy=(x + 0.3, 1.7), xytext=(x, 1.7),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))
    ax.text(5.0, 0.45, r"$\epsilon\sim\mathcal{N}(0,I)$ sampled at every forward pass (train and test)",
            ha="center", fontsize=9)
    ax.set_title("Bayesian Normalization Layer")
    fig.savefig(os.path.join(FIG, "fig_bnl.png"))
    plt.close()


def fig_map_curves():
    m = J("map.json")
    if not m:
        return
    h = m["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3))
    ax[0].plot(ep, [r["train_loss"] for r in h], color="#1F618D", lw=2)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train cross-entropy")
    ax[0].set_title("MAP training loss")
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [r["test_acc"] for r in h], color="#117A65", lw=2)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("test accuracy (%)")
    ax[1].set_title("MAP test accuracy")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_map_curves.png"))
    plt.close()


def fig_abnn_curves():
    a = J("abnn.json")
    if not a:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.3))
    colors = ["#1F618D", "#B03A2E", "#117A65"]
    for run, c in zip(a["runs"], colors):
        ep = [r["epoch"] for r in run["history"]]
        ax[0].plot(ep, [r["train_loss"] for r in run["history"]], color=c, lw=2,
                   label=f"member {run['member']}")
        ax[1].plot(ep, [r["test_acc"] for r in run["history"]], color=c, lw=2,
                   label=f"member {run['member']}")
    ax[0].set_title("ABNN fine-tune loss")
    ax[1].set_title("ABNN fine-tune test acc.")
    for axx, ylab in zip(ax, ["weighted CE", "test accuracy (%)"]):
        axx.set_xlabel("epoch")
        axx.set_ylabel(ylab)
        axx.grid(alpha=0.3)
        axx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_abnn_curves.png"))
    plt.close()


def fig_reliability():
    ev = J("eval.json")
    if not ev:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for ax, key, title in zip(axes, ("map", "mcbn", "abnn"), ("MAP", "MCBN", "ABNN")):
        bins = ev[key]["ece_bins"]
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.bar(bins["bin_conf"], bins["bin_acc"], width=0.06, color="#5DADE2", edgecolor="#1A5276")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}  ECE={ev[key]['ece']:.2f}%")
        ax.set_xlabel("confidence")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_reliability.png"))
    plt.close()


def fig_confidence():
    ev = J("eval.json")
    if not ev:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=True)
    edges = np.linspace(0, 1, 21)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for ax, key, title in zip(axes, ("map", "abnn"), ("MAP", "ABNN")):
        ax.bar(ctr - 0.012, ev[key]["conf_hist"], width=0.022, color="#5DADE2", label="CIFAR-10 (ID)")
        ax.bar(ctr + 0.012, ev[key]["ood"]["conf_hist_ood"], width=0.022, color="#E74C3C", label="SVHN (OOD)")
        ax.set_title(title)
        ax.set_xlabel("max softmax probability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_confidence.png"))
    plt.close()


def fig_metrics():
    ev = J("eval.json")
    if not ev:
        return
    names = ["MAP", "MCBN", "ABNN"]
    keys = ["map", "mcbn", "abnn"]
    metrics = [
        ("Accuracy ↑", [ev[k]["acc"] for k in keys], False),
        ("NLL ↓", [ev[k]["nll"] for k in keys], True),
        ("ECE ↓", [ev[k]["ece"] for k in keys], True),
        ("OOD AUROC ↑", [ev[k]["ood"]["auroc"] for k in keys], False),
        ("OOD FPR95 ↓", [ev[k]["ood"]["fpr95"] for k in keys], True),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 3.2))
    colors = ["#85929E", "#5DADE2", "#1ABC9C"]
    for ax, (title, vals, lower) in zip(axes, metrics):
        bars = ax.bar(names, vals, color=colors, edgecolor="#222", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)
        best = min(vals) if lower else max(vals)
        for b, v in zip(bars, vals):
            if abs(v - best) < 1e-9:
                b.set_linewidth(1.6)
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_metrics.png"))
    plt.close()


def fig_confusion():
    ev = J("eval.json")
    if not ev:
        return
    from data import CLASSES

    mat = np.asarray(ev["abnn"]["confusion"], dtype=float)
    mat = 100.0 * mat / mat.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10), [c[:4] for c in CLASSES], rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    color="white" if mat[i, j] > 55 else "#222", fontsize=7.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("ABNN confusion matrix (row-normalised %)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_confusion.png"))
    plt.close()


def main():
    setup()
    fig_pipeline()
    fig_bnl()
    fig_map_curves()
    fig_abnn_curves()
    fig_reliability()
    fig_confidence()
    fig_metrics()
    fig_confusion()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
