"""Quality checks on a landmark dataset before anyone trains on it.

Hard errors (exit code 1): non-finite values, degenerate hands.
Things to review by eye: class imbalance, low detection rates, and outliers,
which are samples far from the rest of their class and are usually either a
mislabel or a bad landmark fit. Outliers are written to a CSV with their source
so a person can look at the original image and decide.

    python -m fingerspell.validate data/processed/kaggle.npz
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .data import Samples
from .features import normalize, palm_scale


def find_outliers(x: np.ndarray, labels: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Indices of samples whose distance to their class median is more than
    k robust standard deviations above the class's typical distance."""
    flagged = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 10:
            continue
        dist = np.linalg.norm(x[idx] - np.median(x[idx], axis=0), axis=1)
        med = np.median(dist)
        mad = np.median(np.abs(dist - med)) * 1.4826 + 1e-9
        flagged.extend(idx[dist > med + k * mad])
    return np.array(sorted(flagged), dtype=int)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--outlier-k", type=float, default=4.0)
    args = ap.parse_args(argv)

    s = Samples.load(args.data)
    errors = 0
    print(f"{args.data}: {len(s)} samples, {len(np.unique(s.labels))} classes, "
          f"{len(np.unique(s.signer))} signer(s), {len(np.unique(s.group))} groups")

    bad = ~np.isfinite(s.landmarks).all(axis=(1, 2))
    degenerate = ~bad & (palm_scale(s.landmarks) < 1.0)  # palm under one pixel wide
    for name, mask in (("non-finite", bad), ("degenerate", degenerate)):
        if mask.any():
            errors += int(mask.sum())
            print(f"ERROR: {int(mask.sum())} {name} samples, e.g. {s.source[mask][:3].tolist()}")
    ok = ~(bad | degenerate)

    counts = Counter(s.labels.tolist())
    failures = Counter()
    fail_path = args.data.with_suffix(".failures.csv")
    if fail_path.exists():
        with open(fail_path) as fh:
            failures = Counter(row["label"] for row in csv.DictReader(fh))
    print("\nclass     count  left-hand  " + ("detected" if failures else ""))
    for label in sorted(counts):
        m = s.labels == label
        line = f"{label:>8} {counts[label]:6d}  {s.handed[m].mean():8.1%}"
        if failures:
            line += f"  {counts[label] / (counts[label] + failures[label]):8.1%}"
        print(line)
    smallest, largest = min(counts.values()), max(counts.values())
    if largest > 2 * smallest:
        print(f"WARNING: class imbalance, smallest class has {smallest} samples and largest has {largest}")

    x = normalize(s.landmarks[ok], s.handed[ok])
    dup = len(x) - len(np.unique(np.round(x, 4), axis=0))
    if dup:
        print(f"WARNING: {dup} exact duplicate samples")

    out_idx = np.flatnonzero(ok)[find_outliers(x, s.labels[ok], args.outlier_k)]
    review = args.data.with_suffix(".review.csv")
    with open(review, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "signer", "source"])
        writer.writerows(zip(s.labels[out_idx], s.signer[out_idx], s.source[out_idx]))
    print(f"\n{len(out_idx)} outliers ({len(out_idx) / max(len(s), 1):.1%}) listed for review in {review}")
    print("FAILED" if errors else "OK")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
