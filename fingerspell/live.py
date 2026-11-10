"""Live webcam demo.

    python -m fingerspell.live --run runs/kaggle
    python -m fingerspell.live --run runs/kaggle --record demo.mp4

Keys: SPACE add a space | BACKSPACE delete | C clear | R start/stop recording | ESC quit
Hold a letter steady to type it. To type the same letter twice, drop your hand
out of frame briefly and form it again.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .decoder import LetterDecoder
from .tracker import DEFAULT_MODEL_PATH, HAND_CONNECTIONS


def draw_hand(frame: np.ndarray, landmarks: np.ndarray) -> None:
    pts = landmarks[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(map(int, p)), 4, (80, 200, 60), -1, cv2.LINE_AA)


def draw_overlay(view: np.ndarray, label, conf: float, text: str, fps: float, recording: bool) -> None:
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, h - 64), (w, h), (0, 0, 0), -1)
    cv2.putText(view, text[-32:] + "_", (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    if label is not None:
        cv2.putText(view, label, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.rectangle(view, (12, 86), (212, 100), (90, 90, 90), -1)
        cv2.rectangle(view, (12, 86), (12 + int(200 * conf), 100), (80, 200, 60), -1)
    cv2.putText(view, f"{fps:4.1f} fps", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(view, (w - 130, 22), 7, (0, 0, 255), -1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--record", type=Path, default=None, help="write the annotated video here from the start")
    ap.add_argument("--hold-frames", type=int, default=8)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args(argv)

    from .model import Classifier
    from .tracker import HandTracker

    clf = Classifier(args.run)
    decoder = LetterDecoder(clf.classes, min_conf=args.min_conf, hold_frames=args.hold_frames)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    writer, record_path = None, args.record or Path("demo.mp4")
    recording = args.record is not None
    t0 = last = time.time()
    fps = 0.0
    try:
        with HandTracker(args.model, video=True) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                hand = tracker.detect(frame, int((time.time() - t0) * 1000))
                probs = None
                if hand is not None:
                    probs = clf.predict_proba(hand.landmarks, hand.handed)[0]
                    draw_hand(frame, hand.landmarks)
                decoder.update(probs)
                view = cv2.flip(frame, 1)  # mirror for display only; detection used the true image
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                label, conf = decoder.current if hand is not None else (None, 0.0)
                draw_overlay(view, label, conf, decoder.text, fps, recording)
                if recording:
                    if writer is None:
                        writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 20.0, (view.shape[1], view.shape[0]))
                    writer.write(view)
                cv2.imshow("fingerspell", view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord(" "):
                    decoder.space()
                elif key in (8, 127):
                    decoder.backspace()
                elif key in (ord("c"), ord("C")):
                    decoder.clear()
                elif key in (ord("r"), ord("R")):
                    recording = not recording
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[live] saved {record_path}")
        cv2.destroyAllWindows()
        print(f"[live] final text: {decoder.text!r}")


if __name__ == "__main__":
    main()
