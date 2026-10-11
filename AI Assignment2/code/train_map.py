"""Train the deterministic MAP ResNet-20 on CIFAR-10."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import build_cifar_loaders
from resnet import count_parameters, resnet20

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    nll = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        nll += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return {"acc": 100.0 * correct / total, "nll": nll / total}


def train(args):
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)

    train_loader, test_loader = build_cifar_loaders(args.batch_size)
    model = resnet20().to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"MAP ResNet-20 params={count_parameters(model):,} epochs={args.epochs}")

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        ep0 = time.time()
        for x, y in train_loader:
            x = x.to(device).to(memory_format=torch.channels_last)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
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
            "lr": sched.get_last_lr()[0],
            "epoch_time_s": time.time() - ep0,
        }
        history.append(rec)
        print(
            f"MAP {epoch:02d}/{args.epochs}  loss={rec['train_loss']:.4f}  "
            f"acc={rec['test_acc']:.2f}  nll={rec['test_nll']:.4f}  "
            f"{rec['epoch_time_s']:.0f}s"
        )

    ckpt = os.path.join(RES, "map_resnet20.pth")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    out = {
        "epochs": args.epochs,
        "params": count_parameters(model),
        "history": history,
        "final_acc": history[-1]["test_acc"],
        "final_nll": history[-1]["test_nll"],
        "best_acc": max(h["test_acc"] for h in history),
        "total_time_s": time.time() - t0,
        "checkpoint": ckpt,
    }
    with open(os.path.join(RES, "map.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved {ckpt}  best_acc={out['best_acc']:.2f}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
