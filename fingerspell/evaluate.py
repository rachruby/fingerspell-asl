"""Evaluate a trained run.

    python -m fingerspell.evaluate --run runs/kaggle                       # the run's held-out test split
    python -m fingerspell.evaluate --run runs/kaggle --data data/collected/me.npz --name me

Writes metrics JSON and a confusion-matrix PNG into the run directory and
prints per-class precision/recall plus the most frequent confusions, which is
usually the most informative part: it shows which letter pairs the model mixes up.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import Samples
from .model import Classifier


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)  # rows = true class, columns = predicted class
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90 if max(map(len, classes)) > 1 else 0)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data", nargs="+", type=Path, default=None, help="defaults to the run's test.npz")
    ap.add_argument("--name", default="test", help="suffix for output files")
    args = ap.parse_args(argv)

    clf = Classifier(args.run)
    samples = Samples.load_many(args.data or [args.run / "test.npz"])
    known = np.isin(samples.labels, clf.classes)
    if not known.all():
        print(f"[eval] dropping {int((~known).sum())} samples whose label the model was not trained on")
        samples = samples.subset(known)
    if len(samples) == 0:
        raise SystemExit("nothing to evaluate")

    y = np.array([clf.classes.index(label) for label in samples.labels])
    pred = clf.predict_proba(samples.landmarks, samples.handed).argmax(axis=1)
    cm = confusion_matrix(y, pred, len(clf.classes))
    precision, recall, f1 = per_class(cm)
    present = cm.sum(axis=1) > 0

    print(f"\n{'class':>8} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>6}")
    for i, c in enumerate(clf.classes):
        if present[i]:
            print(f"{c:>8} {precision[i]:6.3f} {recall[i]:6.3f} {f1[i]:6.3f} {cm[i].sum():6d}")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = [(int(i), int(j)) for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape)) if off[i, j]]
    if top:
        print("\nmost common confusions (true -> predicted):")
        for i, j in top:
            print(f"  {clf.classes[i]} -> {clf.classes[j]}: {off[i, j]} ({off[i, j] / cm[i].sum():.1%} of {clf.classes[i]})")

    one = samples.landmarks[:1]
    t0 = time.perf_counter()
    for _ in range(200):
        clf.predict_proba(one, samples.handed[:1])
    latency_ms = (time.perf_counter() - t0) / 200 * 1000

    metrics = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1[present].mean()),
        "by_signer": {s: float((pred == y)[samples.signer == s].mean()) for s in np.unique(samples.signer)},
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "n": int(cm[i].sum())}
                      for i, c in enumerate(clf.classes) if present[i]},
        "top_confusions": [{"true": clf.classes[i], "pred": clf.classes[j], "count": int(off[i, j])} for i, j in top],
        "classifier_latency_ms": latency_ms,
    }
    print(f"\naccuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | n={len(y)} | "
          f"classifier {latency_ms:.2f} ms/frame (excludes MediaPipe)")
    for s, a in metrics["by_signer"].items():
        print(f"  signer {s}: {a:.4f}")

    (args.run / f"metrics_{args.name}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, clf.classes, args.run / f"confusion_{args.name}.png",
                   f"{args.name}: accuracy {metrics['accuracy']:.1%}, n={len(y)}")
    print(f"[eval] wrote metrics_{args.name}.json and confusion_{args.name}.png to {args.run}")
    return metrics


if __name__ == "__main__":
    main()
