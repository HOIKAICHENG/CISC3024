"""Supervised training on a labelled CIFAR-10 subset.

With `--init <fcmae checkpoint>` the encoder is initialised from FCMAE
pre-training; without it the identical network is trained from scratch.  Both
paths share the same recipe so that the two runs differ only in initialisation.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from convnextv2 import convnextv2_atto, count_parameters
from data import build_loaders
from engine import RESULTS_DIR, Timer, cosine_lr, evaluate, param_groups_weight_decay, save_json


def load_encoder_weights(model: nn.Module, ckpt_path: str) -> tuple[int, int]:
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    return len(enc_state), len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default="", help="FCMAE checkpoint to start from")
    ap.add_argument("--no-grn", action="store_true")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=3)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        n_per_class=args.n_per_class,
        mode="finetune",
        num_workers=args.workers,
        seed=args.seed,
    )

    model = convnextv2_atto(
        num_classes=10, use_grn=not args.no_grn, drop_path_rate=args.drop_path
    ).to(device, memory_format=torch.channels_last)
    if args.init:
        n_loaded, n_missing = load_encoder_weights(model, args.init)
        print(f"[{args.tag}] loaded {n_loaded} encoder tensors from {os.path.basename(args.init)} "
              f"({n_missing} head tensors left random)", flush=True)
    else:
        print(f"[{args.tag}] random initialisation (from scratch)", flush=True)
    print(f"[{args.tag}] params {count_parameters(model)/1e6:.2f}M | "
          f"train {len(train_loader.dataset)} imgs | threads={torch.get_num_threads()}", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optim = torch.optim.AdamW(
        param_groups_weight_decay(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999)
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    history, best = [], 0.0
    step = 0
    with Timer() as t:
        for epoch in range(args.epochs):
            model.train()
            running, seen, t0 = 0.0, 0, time.time()
            for images, targets in train_loader:
                lr = cosine_lr(step, total_steps, args.lr, warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = lr
                images = images.to(device).contiguous(memory_format=torch.channels_last)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optim.step()
                running += loss.item() * images.size(0)
                seen += images.size(0)
                step += 1
            do_eval = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
            metrics = evaluate(model, test_loader, device) if do_eval else None
            if metrics:
                best = max(best, metrics["acc"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": running / seen,
                    "test_loss": metrics["loss"] if metrics else None,
                    "test_acc": metrics["acc"] if metrics else None,
                    "time_s": round(time.time() - t0, 1),
                }
            )
            acc_str = f"test_acc {metrics['acc']:.2f}%" if metrics else "test_acc  --  "
            print(f"[{args.tag}] epoch {epoch+1}/{args.epochs} train_loss {running/seen:.3f} "
                  f"{acc_str} ({time.time()-t0:.0f}s)", flush=True)

    final = evaluate(model, test_loader, device)
    per_class = []
    for c in range(10):
        m = final["true"] == c
        per_class.append(round(100.0 * (final["pred"][m] == c).sum().item() / m.sum().item(), 2))
    confusion = torch.zeros(10, 10, dtype=torch.int32)
    for p, y in zip(final["pred"].tolist(), final["true"].tolist()):
        confusion[y, p] += 1

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(RESULTS_DIR, f"cls_{args.tag}.pth"))
    save_json(
        f"finetune_{args.tag}.json",
        {
            "tag": args.tag,
            "init": os.path.basename(args.init) if args.init else "random",
            "use_grn": not args.no_grn,
            "params_M": round(count_parameters(model) / 1e6, 3),
            "n_train": len(train_loader.dataset),
            "epochs": args.epochs,
            "lr": args.lr,
            "final_acc": round(final["acc"], 2),
            "best_acc": round(best, 2),
            "per_class_acc": per_class,
            "confusion": confusion.tolist(),
            "total_time_s": round(t.elapsed, 1),
            "history": history,
        },
    )
    print(f"[{args.tag}] final {final['acc']:.2f}% | best {best:.2f}% | {t.elapsed/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
