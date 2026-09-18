"""Post-hoc analysis of the FCMAE checkpoints.

1. Feature-collapse diagnostic (Sec. 3 of the ConvNeXt V2 paper): average
   pairwise cosine distance between channel activations inside every block's
   MLP, for the V1 (no GRN) and V2 (GRN) encoders.
2. k-NN accuracy on frozen, globally pooled encoder features: how good is the
   representation *before* any supervised fine-tuning.
3. Qualitative masked-image reconstructions of the V2 model.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from convnextv2 import Block
from data import CIFAR10Local, DATA_ROOT, denormalize, eval_transform
from engine import RESULTS_DIR
from fcmae import FCMAE

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_fcmae(path: str) -> FCMAE:
    ckpt = torch.load(path, map_location="cpu")
    model = FCMAE(mask_ratio=ckpt["args"]["mask_ratio"], use_grn=not ckpt["args"]["no_grn"])
    model.load_state_dict(ckpt["model"])
    return model.eval()


# --------------------------------------------------------------------------
# 1. feature cosine distance
# --------------------------------------------------------------------------
@torch.no_grad()
def channel_cosine_distance(model: FCMAE, images: torch.Tensor):
    """For every encoder block return mean over images of
    1/C^2 * sum_ij (1 - cos(X_i, X_j)) / 2, computed on the MLP hidden features
    (input of pwconv2), i.e. after GELU (V1) or after GRN (V2)."""
    captured = []
    hooks = []
    for stage in model.encoder.stages:
        for blk in stage:
            assert isinstance(blk, Block)
            hooks.append(blk.pwconv2.register_forward_pre_hook(
                lambda m, inp: captured.append(inp[0].detach())
            ))
    model.encoder.forward_features(images, mask=None)
    for h in hooks:
        h.remove()

    dists = []
    for feat in captured:  # (N, H, W, 4C)
        n, h, w, c = feat.shape
        x = feat.permute(0, 3, 1, 2).reshape(n, c, h * w)
        x = F.normalize(x, dim=-1, eps=1e-8)
        cos = torch.bmm(x, x.transpose(1, 2))  # (N, C, C)
        dist = ((1.0 - cos) / 2.0).mean(dim=(1, 2))  # (N,)
        dists.append(dist.mean().item())
    return dists


# --------------------------------------------------------------------------
# 2. k-NN on frozen features
# --------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model: FCMAE, loader: DataLoader):
    feats, labels = [], []
    for images, targets in loader:
        images = images.contiguous(memory_format=torch.channels_last)
        x, _ = model.encoder.forward_features(images, mask=None)
        feats.append(F.normalize(x.mean([-2, -1]), dim=-1))
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(train_f, train_y, test_f, test_y, k: int = 20, temperature: float = 0.07):
    correct = 0
    for i in range(0, test_f.shape[0], 512):
        sim = test_f[i:i + 512] @ train_f.T  # cosine similarity
        topv, topi = sim.topk(k, dim=1)
        neigh = train_y[topi]  # (b, k)
        weights = (topv / temperature).exp()
        votes = torch.zeros(sim.shape[0], 10).scatter_add_(1, neigh, weights)
        correct += (votes.argmax(1) == test_y[i:i + 512]).sum().item()
    return 100.0 * correct / test_f.shape[0]


# --------------------------------------------------------------------------
# 3. reconstructions
# --------------------------------------------------------------------------
@torch.no_grad()
def reconstruction_figure(model: FCMAE, images: torch.Tensor, path: str, seed: int = 0):
    torch.manual_seed(seed)
    loss, pred, mask = model(images)
    target = model.patchify(images)
    mean = target.mean(dim=-1, keepdim=True)
    std = (target.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
    pred_px = model.unpatchify(pred * std + mean)  # undo per-patch normalisation

    m_img = model.unpatchify(mask.flatten(1).unsqueeze(-1).expand(-1, -1, target.shape[-1]))
    masked = images * (1 - m_img)
    pasted = images * (1 - m_img) + pred_px * m_img

    rows = [images, masked, pred_px, pasted]
    titles = ["original", "masked input (60%)", "prediction", "visible + prediction"]
    n = images.shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.3 * n, 1.45 * len(rows)))
    for r, (imgs, title) in enumerate(zip(rows, titles)):
        imgs = denormalize(imgs)
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
    fig.suptitle(f"FCMAE reconstruction on CIFAR-10 test images (loss on masked patches = {loss.item():.3f})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=os.path.join(RESULTS_DIR, "fcmae_v2_grn.pth"))
    ap.add_argument("--v1", default=os.path.join(RESULTS_DIR, "fcmae_v1_nogrn.pth"))
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--knn-train", type=int, default=50000)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(FIG_DIR, exist_ok=True)

    models = {"ConvNeXt V2 (GRN)": load_fcmae(args.v2), "ConvNeXt V1 (no GRN)": load_fcmae(args.v1)}

    test_ds = CIFAR10Local(DATA_ROOT, train=False, transform=eval_transform())
    train_ds = CIFAR10Local(DATA_ROOT, train=True, transform=eval_transform())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    if args.knn_train < len(train_ds):
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train_ds), generator=g)[: args.knn_train]
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)

    out = {"cosine_distance": {}, "knn_acc": {}, "block_labels": []}

    # ---- collapse diagnostic on 512 test images
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(test_ds), generator=g)[:512]
    probe = torch.stack([test_ds[i][0] for i in idx.tolist()])
    for name, model in models.items():
        out["cosine_distance"][name] = channel_cosine_distance(model, probe)
        print(f"{name}: cosine distance per block = "
              f"{[round(d, 3) for d in out['cosine_distance'][name]]}", flush=True)
    depths = models["ConvNeXt V2 (GRN)"].encoder.depths
    for s, d in enumerate(depths):
        out["block_labels"] += [f"S{s+1}.{b+1}" for b in range(d)]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    xs = range(1, len(out["block_labels"]) + 1)
    for (name, dists), mk in zip(out["cosine_distance"].items(), ("o", "s")):
        ax.plot(xs, dists, marker=mk, label=name)
    ax.set_xticks(list(xs)); ax.set_xticklabels(out["block_labels"], rotation=45, fontsize=7)
    ax.set_xlabel("encoder block (stage.block)"); ax.set_ylabel("mean channel cosine distance")
    ax.set_title("Feature diversity inside the MLP of each block after FCMAE pre-training", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig_feature_collapse.png"), dpi=200)
    plt.close(fig)

    # ---- k-NN on frozen features
    for name, model in models.items():
        tr_f, tr_y = extract_features(model, train_loader)
        te_f, te_y = extract_features(model, test_loader)
        acc = knn_accuracy(tr_f, tr_y, te_f, te_y)
        out["knn_acc"][name] = round(acc, 2)
        print(f"{name}: 20-NN accuracy on frozen features = {acc:.2f}%", flush=True)

    # ---- reconstructions
    vis = torch.stack([test_ds[i][0] for i in range(0, 80, 10)])
    out["recon_loss_vis"] = reconstruction_figure(
        models["ConvNeXt V2 (GRN)"], vis, os.path.join(FIG_DIR, "fig_reconstruction.png")
    )

    with open(os.path.join(RESULTS_DIR, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved analysis.json and figures", flush=True)


if __name__ == "__main__":
    main()
