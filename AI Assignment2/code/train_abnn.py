"""Fine-tune ABNN members from the MAP checkpoint.

Only BNL affine parameters (gamma, beta) are trained.  Each member gets its
own class-dependent loss weights so the ensemble explores different posterior
modes, as in Section 4.2 of the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bnl import count_bnl, freeze_except_bnl, replace_bn_with_bnl
from data import CLASSES, build_cifar_loaders
from resnet import resnet20

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def class_weights(seed: int, n_classes: int = 10, n_prioritized: int = 4, factor: float = 5.0):
    rng = np.random.RandomState(seed)
    w = np.ones(n_classes, dtype=np.float32)
    chosen = rng.choice(n_classes, size=n_prioritized, replace=False)
    w[chosen] = factor
    return torch.tensor(w), [CLASSES[i] for i in chosen]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    nll = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # One stochastic pass is enough for the training-time monitor.
        logits = model(x)
        nll += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return {"acc": 100.0 * correct / total, "nll": nll / total}


def load_map_as_abnn(ckpt_path: str, device, noise_std: float = 0.1):
    model = resnet20()
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    replace_bn_with_bnl(model, noise_std=noise_std)
    model.to(device)
    model.to(memory_format=torch.channels_last)
    trainable = freeze_except_bnl(model)
    return model, trainable


def train_member(member: int, args, train_loader, test_loader, device):
    model, trainable = load_map_as_abnn(args.ckpt, device, noise_std=args.noise_std)
    from bnl import set_bnl_stochastic

    set_bnl_stochastic(model, False)
    frozen = evaluate(model, test_loader, device)
    set_bnl_stochastic(model, True)
    print(f"  conversion check (ε=0): acc={frozen['acc']:.2f}  (should stay near the MAP accuracy)")
    weights, experts = class_weights(
        seed=args.seed + 17 * (member + 1),
        n_prioritized=args.n_prioritized,
        factor=args.factor,
    )
    weights = weights.to(device)
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    history = []
    t0 = time.time()
    print(
        f"ABNN member {member}: experts={experts}  "
        f"trainable={sum(p.numel() for p in trainable):,}  BNL={count_bnl(model)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        ep0 = time.time()
        for x, y in train_loader:
            x = x.to(device).to(memory_format=torch.channels_last)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            # L = L_MAP + E  <=>  weighted CE with class weight (1 + eta) = paper factor.
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        sched.step()
        metrics = evaluate(model, test_loader, device)
        rec = {
            "epoch": epoch,
            "train_loss": running / seen,
            "test_acc": metrics["acc"],
            "test_nll": metrics["nll"],
            "epoch_time_s": time.time() - ep0,
        }
        history.append(rec)
        print(
            f"  m{member} {epoch:02d}/{args.epochs}  loss={rec['train_loss']:.4f}  "
            f"acc={rec['test_acc']:.2f}  {rec['epoch_time_s']:.0f}s"
        )

    ckpt = os.path.join(RES, f"abnn_member{member}.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "experts": experts,
            "class_weights": weights.cpu().tolist(),
            "member": member,
        },
        ckpt,
    )
    return {
        "member": member,
        "experts": experts,
        "class_weights": weights.cpu().tolist(),
        "history": history,
        "final_acc": history[-1]["test_acc"],
        "best_acc": max(h["test_acc"] for h in history),
        "total_time_s": time.time() - t0,
        "checkpoint": ckpt,
        "n_bnl": count_bnl(model),
        "n_trainable": sum(p.numel() for p in trainable),
    }


def train(args):
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_loader, test_loader = build_cifar_loaders(args.batch_size)

    members = []
    t0 = time.time()
    for m in range(args.members):
        members.append(train_member(m, args, train_loader, test_loader, device))

    out = {
        "members": args.members,
        "epochs": args.epochs,
        "factor": args.factor,
        "n_prioritized": args.n_prioritized,
        "noise_std": args.noise_std,
        "runs": members,
        "total_time_s": time.time() - t0,
    }
    with open(os.path.join(RES, "abnn.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"ABNN done in {out['total_time_s']/60:.1f} min")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=os.path.join(RES, "map_resnet20.pth"))
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--factor", type=float, default=5.0)
    p.add_argument("--n-prioritized", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
