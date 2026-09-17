"""Shared training utilities: LR schedule, parameter groups, evaluation."""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn as nn

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float = 1e-6):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def param_groups_weight_decay(model: nn.Module, weight_decay: float):
    """No weight decay on biases, norms, GRN affine parameters and mask token."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "mask_token" in name or "grn" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str = "cpu"):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_pred, all_true = [], []
    for images, targets in loader:
        images = images.to(device).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device)
        logits = model(images)
        loss_sum += criterion(logits, targets).item()
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
        all_pred.append(pred.cpu())
        all_true.append(targets.cpu())
    return {
        "acc": 100.0 * correct / total,
        "loss": loss_sum / total,
        "pred": torch.cat(all_pred),
        "true": torch.cat(all_true),
    }


def save_json(name: str, payload: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
