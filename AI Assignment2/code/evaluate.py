"""Evaluate MAP, MCBN, MC-Dropout-style sampling, and ABNN.

Metrics follow the paper: accuracy, NLL, ECE, and OOD detection of CIFAR-10
vs SVHN using maximum softmax probability (AUPR, AUROC, FPR95).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from bnl import replace_bn_with_bnl, set_bnl_stochastic
from data import CLASSES, build_cifar_loaders, build_svhn_loader
from resnet import resnet20

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1]) if i > 0 else (conf >= bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc = correct[mask].mean()
            c = conf[mask].mean()
            ece += mask.mean() * abs(acc - c)
            bin_acc.append(float(acc))
            bin_conf.append(float(c))
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.5 * (bins[i] + bins[i + 1]))
        bin_count.append(int(mask.sum()))
    return float(ece), {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count}


def nll_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(100.0 * (probs.argmax(1) == labels).mean())


def roc_auc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    y = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    s = np.concatenate([scores_id, scores_ood])
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area(tpr, fpr))


def pr_auc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    y = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    s = np.concatenate([scores_id, scores_ood])
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / max(tps[-1], 1)
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area(prec, rec))


def fpr95(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    thr = np.quantile(scores_id, 0.05)
    return float(100.0 * (scores_ood >= thr).mean())


def collect_logits(model, loader, device, n_samples: int = 1, train_bn: bool = False):
    """Average ``n_samples`` softmaxes.  ``train_bn`` implements the MCBN baseline."""
    if train_bn:
        model.train()
    else:
        model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            acc = 0.0
            for _ in range(n_samples):
                acc = acc + F.softmax(model(x), dim=1)
            all_probs.append((acc / n_samples).cpu())
            all_y.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_y).numpy()


def collect_abnn(ckpts: list[str], loader, device, l_samples: int, noise_std: float = 0.1):
    probs_acc = None
    labels = None
    member_probs = []
    for path in ckpts:
        model = resnet20()
        replace_bn_with_bnl(model, noise_std=noise_std)
        blob = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        model.to(device)
        model.eval()
        set_bnl_stochastic(model, True)
        p, y = collect_logits(model, loader, device, n_samples=l_samples)
        member_probs.append(p)
        probs_acc = p if probs_acc is None else probs_acc + p
        labels = y
    return probs_acc / len(ckpts), labels, member_probs


def disagreement(member_probs: list[np.ndarray]) -> float:
    preds = np.stack([p.argmax(1) for p in member_probs], axis=0)
    n = 0
    d = 0
    m = preds.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            d += (preds[i] != preds[j]).mean()
            n += 1
    return float(d / max(n, 1))


def pack(name, probs, labels, ood_probs=None):
    ece, bins = ece_score(probs, labels)
    out = {
        "name": name,
        "acc": accuracy(probs, labels),
        "nll": nll_score(probs, labels),
        "ece": 100.0 * ece,
        "ece_bins": bins,
        "confidence": None,
        "conf_mean": float(probs.max(1).mean()),
        "conf_hist": np.histogram(probs.max(1), bins=20, range=(0, 1))[0].tolist(),
        "confusion": _confusion(probs, labels),
        "per_class_acc": _per_class(probs, labels),
    }
    if ood_probs is not None:
        sid, sood = probs.max(1), ood_probs.max(1)
        out["ood"] = {
            "auroc": 100.0 * roc_auc(sid, sood),
            "aupr": 100.0 * pr_auc(sid, sood),
            "fpr95": fpr95(sid, sood),
            "conf_mean_ood": float(sood.mean()),
            "conf_hist_ood": np.histogram(sood, bins=20, range=(0, 1))[0].tolist(),
        }
    return out


def _confusion(probs, labels):
    pred = probs.argmax(1)
    mat = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(labels, pred):
        mat[t, p] += 1
    return mat.tolist()


def _per_class(probs, labels):
    pred = probs.argmax(1)
    out = {}
    for c, name in enumerate(CLASSES):
        mask = labels == c
        out[name] = float(100.0 * (pred[mask] == c).mean()) if mask.any() else 0.0
    return out


def evaluate(args):
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    _, id_loader = build_cifar_loaders(args.batch_size)
    ood_loader = build_svhn_loader(args.batch_size)

    map_model = resnet20()
    map_blob = torch.load(args.map_ckpt, map_location=device, weights_only=False)
    map_model.load_state_dict(map_blob["model"])
    map_model.to(device)

    print("evaluating MAP ...")
    p_map, y = collect_logits(map_model, id_loader, device, 1)
    o_map, _ = collect_logits(map_model, ood_loader, device, 1)

    print("evaluating MCBN ...")
    p_mcbn, _ = collect_logits(map_model, id_loader, device, args.l_samples, train_bn=True)
    o_mcbn, _ = collect_logits(map_model, ood_loader, device, args.l_samples, train_bn=True)

    abnn_ckpts = [os.path.join(RES, f"abnn_member{m}.pth") for m in range(args.members)]
    print(f"evaluating ABNN M={args.members} L={args.l_samples} ...")
    p_abnn, _, member_p = collect_abnn(abnn_ckpts, id_loader, device, args.l_samples, args.noise_std)
    o_abnn, _, _ = collect_abnn(abnn_ckpts, ood_loader, device, args.l_samples, args.noise_std)

    results = {
        "map": pack("MAP (single model)", p_map, y, o_map),
        "mcbn": pack("MCBN", p_mcbn, y, o_mcbn),
        "abnn": pack("ABNN", p_abnn, y, o_abnn),
        "abnn_disagreement": disagreement(member_p),
        "members": args.members,
        "l_samples": args.l_samples,
        "n_id": int(len(y)),
        "n_ood": int(len(o_map)),
    }
    # keep a small sample of softmax vectors for the reliability / example figure
    rng = np.random.RandomState(0)
    idx = rng.choice(len(y), size=min(32, len(y)), replace=False)
    results["examples"] = {
        "idx": idx.tolist(),
        "y": y[idx].tolist(),
        "map_prob": p_map[idx].tolist(),
        "abnn_prob": p_abnn[idx].tolist(),
    }
    path = os.path.join(RES, "eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("ece_bins", "confidence", "conf_hist", "confusion")}
                      if isinstance(v, dict) else v
                      for k, v in results.items() if k != "examples"}, indent=2))
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map-ckpt", type=str, default=os.path.join(RES, "map_resnet20.pth"))
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--l-samples", type=int, default=3)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
