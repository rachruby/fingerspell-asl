"""Train the landmark classifier.

    python -m fingerspell.train --data data/processed/kaggle.npz --out runs/kaggle

Pass several --data files to pool them. With --holdout-signer NAME, every sample
from that signer is kept out of training and becomes the test set, which is the
number that tells you whether the model works on a person it has never seen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Samples
from .features import normalize
from .model import LandmarkMLP, augment, save_checkpoint
from .splits import group_split


@torch.no_grad()
def accuracy(model, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean()) if len(x) else float("nan")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=[], help="labels to drop, e.g. J Z")
    ap.add_argument("--holdout-signer", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 128])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without val improvement")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    samples = Samples.load_many(args.data)
    samples = samples.subset(~np.isin(samples.labels, args.exclude))

    if args.holdout_signer:
        held = samples.signer == args.holdout_signer
        if not held.any():
            raise SystemExit(f"no samples from signer {args.holdout_signer!r}")
        split = np.full(len(samples), "test", dtype="U5")
        split[~held] = group_split(samples.labels[~held], samples.group[~held], val=0.15, test=0.0, seed=args.seed)
    else:
        split = group_split(samples.labels, samples.group, seed=args.seed)

    classes = sorted(np.unique(samples.labels[split == "train"]).tolist())
    known = np.isin(samples.labels, classes)
    to_index = {c: i for i, c in enumerate(classes)}

    def tensors(name):
        part = samples.subset((split == name) & known)
        x = torch.from_numpy(normalize(part.landmarks, part.handed))
        y = torch.tensor([to_index[label] for label in part.labels], dtype=torch.long)
        return x, y

    (xtr, ytr), (xva, yva), (xte, yte) = tensors("train"), tensors("val"), tensors("test")
    print(f"[train] {len(classes)} classes | train {len(xtr)}  val {len(xva)}  test {len(xte)}")

    args.out.mkdir(parents=True, exist_ok=True)
    samples.subset((split == "test") & known).save(args.out / "test.npz")

    model = LandmarkMLP(len(classes), args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best, stale, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order, total = torch.randperm(len(xtr)), 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            if len(idx) < 2:  # BatchNorm cannot run on a single sample
                continue
            xb = xtr[idx] if args.no_augment else augment(xtr[idx])
            loss = loss_fn(model(xb), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_acc = accuracy(model, xva, yva)
        history.append({"epoch": epoch, "loss": total / len(xtr), "val_acc": val_acc})
        print(f"[train] epoch {epoch:3d}  loss {total / len(xtr):.4f}  val acc {val_acc:.4f}")
        score = val_acc if len(xva) else -total  # no val set: fall back to training loss
        if score > best:
            best, stale = score, 0
            save_checkpoint(args.out / "model.pt", model, classes, args.hidden, args.dropout)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[train] early stop at epoch {epoch}")
                break

    summary = {"classes": classes, "n_train": len(xtr), "n_val": len(xva), "n_test": len(xte),
               "best_val_acc": best if len(xva) else None, "seconds": round(time.time() - t0, 1),
               "args": {k: (str(v) if isinstance(v, Path) else [str(i) for i in v] if k == "data" else v)
                        for k, v in vars(args).items()},
               "history": history}
    (args.out / "train.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] saved {args.out / 'model.pt'}; next: python -m fingerspell.evaluate --run {args.out}")
    return summary


if __name__ == "__main__":
    main()
