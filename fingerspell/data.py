"""The one on-disk format every tool in this repo reads and writes.

A dataset is an .npz of parallel arrays, one row per hand observation. We store
*raw* landmarks (pixel units) rather than normalized features so the feature
code can change without re-running extraction, and we store no images, which
keeps self-collected data small and less privacy-sensitive.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

N_LANDMARKS = 21


@dataclass
class Samples:
    landmarks: np.ndarray  # (N, 21, 3) float32. x, y in pixels; z in the same scale, relative to wrist
    handed: np.ndarray     # (N,) int8. 1 if MediaPipe labelled the hand "Left", else 0
    labels: np.ndarray     # (N,) str. Class name, e.g. "A"
    signer: np.ndarray     # (N,) str. Who produced the sample
    source: np.ndarray     # (N,) str. Image path or frame id, for tracing a sample back
    group: np.ndarray      # (N,) str. Samples sharing a group are near-duplicates and must not be split

    def __post_init__(self):
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32).reshape(-1, N_LANDMARKS, 3)
        self.handed = np.asarray(self.handed, dtype=np.int8).reshape(-1)
        for name in ("labels", "signer", "source", "group"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=str).reshape(-1))
        n = len(self.landmarks)
        for f in fields(self):
            if len(getattr(self, f.name)) != n:
                raise ValueError(f"{f.name} has {len(getattr(self, f.name))} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.landmarks)

    def subset(self, idx) -> "Samples":
        return Samples(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @staticmethod
    def load(path) -> "Samples":
        with np.load(path, allow_pickle=False) as z:
            return Samples(**{f.name: z[f.name] for f in fields(Samples)})

    @staticmethod
    def concat(parts: list["Samples"]) -> "Samples":
        return Samples(**{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(Samples)})

    @staticmethod
    def load_many(paths) -> "Samples":
        return Samples.concat([Samples.load(p) for p in paths])
