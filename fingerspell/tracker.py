"""Thin wrapper around MediaPipe's HandLandmarker (Tasks API).

Everything MediaPipe-specific lives here, so swapping the landmark model
later means changing one file.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")

# Pairs of landmark indices to draw as the hand skeleton.
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 3) float32, pixel units
    handed: int            # 1 if MediaPipe says "Left"
    score: float


def ensure_model(path=DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[tracker] downloading hand landmark model to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, video: bool = False, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._video = video
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray, timestamp_ms: int | None = None) -> Hand | None:
        """Find one hand in a BGR image (OpenCV's channel order)."""
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        # MediaPipe returns x, y as fractions of width/height. Convert to pixels
        # so a non-square frame does not stretch the hand. z shares x's scale.
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in result.hand_landmarks[0]], dtype=np.float32)
        category = result.handedness[0][0]
        return Hand(pts, int(category.category_name == "Left"), float(category.score))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
