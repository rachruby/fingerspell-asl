"""Train/val/test assignment that respects groups.

The public ASL alphabet sets are consecutive video frames of one person, so
frame 1001 and frame 1002 are almost the same picture. A random split puts one
in train and the other in test and reports ~100% accuracy that means nothing.
Splitting whole groups (blocks of consecutive frames, or one recording burst)
keeps near-duplicates on the same side.
"""
from __future__ import annotations

import numpy as np


def group_split(labels, groups, val: float = 0.15, test: float = 0.15, seed: int = 0) -> np.ndarray:
    """Return an array of "train" / "val" / "test", one per sample.

    Stratified by label: each class's groups are shuffled and divided
    separately, so every class appears in every split when it has enough groups.
    """
    labels = np.asarray(labels, dtype=str)
    groups = np.asarray(groups, dtype=str)
    rng = np.random.default_rng(seed)
    out = np.full(len(labels), "train", dtype="U5")
    for label in np.unique(labels):
        in_class = labels == label
        class_groups = np.unique(groups[in_class])
        rng.shuffle(class_groups)
        n = len(class_groups)
        n_test = max(1, round(n * test)) if test > 0 and n >= 3 else 0
        n_val = max(1, round(n * val)) if val > 0 and n - n_test >= 2 else 0
        if (test > 0 and n_test == 0) or (val > 0 and n_val == 0):
            print(f"[split] warning: class {label!r} has only {n} group(s); some splits will lack it")
        out[in_class & np.isin(groups, class_groups[:n_test])] = "test"
        out[in_class & np.isin(groups, class_groups[n_test:n_test + n_val])] = "val"
    return out
