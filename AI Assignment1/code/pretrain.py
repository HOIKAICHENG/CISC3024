"""FCMAE self-supervised pre-training on the unlabeled CIFAR-10 train split."""

from __future__ import annotations

import argparse
import os
import time

import torch

from data import build_loaders
from engine import RESULTS_DIR, Timer, cosine_lr, param_groups_weight_decay, save_json
from fcmae import FCMAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grn", action="store_true", help="use V1 blocks (LayerScale, no GRN)")
    ap.add_argument("--tag", type=str, default="v2_grn")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit-batches", type=int, default=0, help="0 = full epoch (smoke tests)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _ = build_loaders(
        batch_size=args.batch_size, mode="pretrain", num_workers=args.workers
    )

    # channels_last is ~25% faster than contiguous for these convolutions on CPU
    model = FCMAE(mask_ratio=args.mask_ratio, use_grn=not args.no_grn).to(
        device, memory_format=torch.channels_last
    )
    n_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"[{args.tag}] encoder params: {n_params/1e6:.2f}M | device={device} | "
          f"threads={torch.get_num_threads()}", flush=True)

    optim = torch.optim.AdamW(
        param_groups_weight_decay(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )
    steps_per_epoch = args.limit_batches or len(loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    history = []
    step = 0
    with Timer() as t:
        for epoch in range(args.epochs):
            model.train()
            running, seen, t0 = 0.0, 0, time.time()
            for i, (images, _) in enumerate(loader):
                if args.limit_batches and i >= args.limit_batches:
                    break
                lr = cosine_lr(step, total_steps, args.lr, warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = lr
                images = images.to(device).contiguous(memory_format=torch.channels_last)
                loss, _, _ = model(images)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optim.step()
                running += loss.item() * images.size(0)
                seen += images.size(0)
                step += 1
            epoch_loss = running / seen
            history.append({"epoch": epoch + 1, "loss": epoch_loss, "lr": lr,
                            "time_s": round(time.time() - t0, 1)})
            print(f"[{args.tag}] epoch {epoch+1}/{args.epochs} "
                  f"loss {epoch_loss:.4f} lr {lr:.2e} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt = os.path.join(RESULTS_DIR, f"fcmae_{args.tag}.pth")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
    save_json(
        f"pretrain_{args.tag}.json",
        {
            "tag": args.tag,
            "use_grn": not args.no_grn,
            "encoder_params_M": round(n_params / 1e6, 3),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mask_ratio": args.mask_ratio,
            "total_time_s": round(t.elapsed, 1),
            "history": history,
        },
    )
    print(f"[{args.tag}] done in {t.elapsed/60:.1f} min -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
