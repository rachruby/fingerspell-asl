"""Turns a stream of per-frame probabilities into typed text.

A per-frame classifier flickers, and the hand passes through other letters'
shapes while moving between two letters. The decoder smooths probabilities
over time and only commits a letter once it has been held steadily.
"""
from __future__ import annotations

import numpy as np

SPACE_LABELS = {"space"}
DELETE_LABELS = {"del", "delete"}


class LetterDecoder:
    def __init__(self, classes: list[str], ema: float = 0.5, min_conf: float = 0.6,
                 hold_frames: int = 8, release_frames: int = 5):
        self.classes = list(classes)
        self.ema = ema                        # weight on history; higher = smoother but laggier
        self.min_conf = min_conf              # ignore frames where the top class is below this
        self.hold_frames = hold_frames        # consecutive confident frames needed to commit
        self.release_frames = release_frames  # frames without a hand before the same letter can repeat
        self.text = ""
        self._p = None
        self._candidate = None
        self._streak = 0
        self._armed = True
        self._missing = 0

    @property
    def current(self) -> tuple[str | None, float]:
        """Smoothed top prediction right now, whether or not it was committed."""
        if self._p is None:
            return None, 0.0
        i = int(np.argmax(self._p))
        return self.classes[i], float(self._p[i])

    def update(self, probs) -> str | None:
        """Feed one frame (None when no hand was found). Returns a label on the
        frame it is committed, otherwise None."""
        if probs is None:
            self._missing += 1
            if self._missing >= self.release_frames:
                self._p, self._candidate, self._streak, self._armed = None, None, 0, True
            return None
        self._missing = 0
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        self._p = probs if self._p is None else self.ema * self._p + (1 - self.ema) * probs
        label, conf = self.current
        if conf < self.min_conf:
            self._streak = 0  # keep candidate and armed state so a flicker cannot double-commit
            return None
        if label != self._candidate:
            self._candidate, self._streak, self._armed = label, 1, True
        else:
            self._streak += 1
        if self._armed and self._streak >= self.hold_frames:
            self._armed = False
            self._apply(label)
            return label
        return None

    def _apply(self, label: str) -> None:
        if label in SPACE_LABELS:
            self.text += " "
        elif label in DELETE_LABELS:
            self.text = self.text[:-1]
        else:
            self.text += label

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def space(self) -> None:
        self.text += " "

    def clear(self) -> None:
        self.text = ""
