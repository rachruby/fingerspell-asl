"""Landmarks -> model input.

What we normalize away: where the hand is in the frame, how big it is, and
which hand it is. What we deliberately keep: orientation. In ASL fingerspelling
orientation is contrastive (G vs Q and K vs P are the same handshape pointed in
different directions; H is roughly U turned sideways), so rotating every hand
to a canonical pose would erase real distinctions.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
MCPS = [5, 9, 13, 17]  # index, middle, ring, pinky knuckles
N_FEATURES = 63


def palm_scale(landmarks: np.ndarray) -> np.ndarray:
    """Mean wrist-to-knuckle distance. Unlike a bounding box, this barely
    changes between a fist and an open hand."""
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 21, 3)
    return np.linalg.norm(lm[:, MCPS] - lm[:, WRIST:WRIST + 1], axis=-1).mean(axis=1)


def normalize(landmarks: np.ndarray, handed: np.ndarray) -> np.ndarray:
    """(N, 21, 3) raw landmarks + (N,) handedness -> (N, 63) features."""
    lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 21, 3).copy()
    handed = np.asarray(handed).reshape(-1)
    scale = np.maximum(palm_scale(lm), 1e-6)
    lm -= lm[:, WRIST:WRIST + 1]
    lm[:, :, 0] *= np.where(handed == 1, -1.0, 1.0)[:, None]  # mirror left hands onto right
    lm /= scale[:, None, None]
    return lm.reshape(len(lm), N_FEATURES)
