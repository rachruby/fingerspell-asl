"""The classifier, its training-time augmentation, and an inference wrapper."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import N_FEATURES, normalize


class LandmarkMLP(nn.Module):
    """63 numbers in, one logit per letter out. About 50k parameters, so it
    trains in a minute on a laptop CPU and adds well under a millisecond per frame."""

    def __init__(self, n_classes: int, hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        layers, width = [], N_FEATURES
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def augment(x: torch.Tensor, max_rot_deg: float = 12.0, scale: float = 0.08, noise: float = 0.015) -> torch.Tensor:
    """Small in-plane rotation, scale jitter and landmark noise.

    The rotation range is kept small on purpose: orientation distinguishes
    letters (see features.py), so large rotations would relabel the data.
    """
    b = x.shape[0]
    pts = x.view(b, 21, 3)
    theta = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(max_rot_deg)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(b, 3, 3, device=x.device)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1.0
    pts = pts @ rot.transpose(1, 2)
    pts = pts * (1 + (torch.rand(b, 1, 1, device=x.device) * 2 - 1) * scale)
    pts = pts + torch.randn_like(pts) * noise
    return pts.reshape(b, -1)


def save_checkpoint(path, model: LandmarkMLP, classes: list[str], hidden, dropout: float) -> None:
    torch.save({"state_dict": model.state_dict(), "classes": list(classes),
                "hidden": list(hidden), "dropout": dropout}, path)


class Classifier:
    """Loads a run directory and maps raw landmarks to class probabilities."""

    def __init__(self, run_dir):
        ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=True)
        self.classes: list[str] = ckpt["classes"]
        self.model = LandmarkMLP(len(self.classes), ckpt["hidden"], ckpt["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, landmarks: np.ndarray, handed) -> np.ndarray:
        """(N, 21, 3) or (21, 3) raw landmarks -> (N, C) probabilities."""
        lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 21, 3)
        handed = np.broadcast_to(np.asarray(handed).reshape(-1), (len(lm),))
        x = torch.from_numpy(normalize(lm, handed))
        return torch.softmax(self.model(x), dim=1).numpy()
