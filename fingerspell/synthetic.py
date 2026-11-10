"""Fake landmark data for smoke-testing the pipeline without a camera or a download.

Each class is a random 21-point "hand" plus noise, placed at a random position
and size, half of them mirrored as left hands. It checks that the code runs end
to end; the accuracy it produces says nothing about real signing.

    python -m fingerspell.synthetic --out data/processed/synth.npz
"""
from __future__ import annotations

import argparse
import string
from pathlib import Path

import numpy as np

from .data import Samples


def make_synthetic(classes=tuple(string.ascii_uppercase), per_class: int = 200, signer: str = "synthetic",
                   noise: float = 4.0, block: int = 25, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    prototypes = {c: rng.normal(0, 60, size=(21, 3)) for c in classes}  # shared across signers
    rng = np.random.default_rng([seed, sum(map(ord, signer))])
    rows = {k: [] for k in ("landmarks", "handed", "labels", "signer", "source", "group")}
    for c in classes:
        for i in range(per_class):
            pts = prototypes[c] + rng.normal(0, noise, size=(21, 3))
            handed = int(rng.random() < 0.5)
            if handed:
                pts[:, 0] *= -1
            pts = pts * rng.uniform(0.5, 2.0)
            pts[:, :2] += rng.uniform(100, 500, size=2)
            rows["landmarks"].append(pts)
            rows["handed"].append(handed)
            rows["labels"].append(c)
            rows["signer"].append(signer)
            rows["source"].append(f"{signer}/{c}/{i}")
            rows["group"].append(f"{signer}/{c}/{i // block}")
    return Samples(**rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--signer", default="synthetic")
    a = ap.parse_args()
    data = make_synthetic(per_class=a.per_class, signer=a.signer)
    data.save(a.out)
    print(f"wrote {len(data)} synthetic samples to {a.out}")
