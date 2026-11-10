"""Record your own labelled samples from the webcam.

Press a letter key, then hold that handshape while the tool records a burst of
frames; move your hand around a little so the burst is not 60 copies of one
pose. Only landmarks are saved, never video. Each burst becomes one split group.

    python -m fingerspell.collect --signer rachel --out data/collected/rachel.npz

Keys: a-z record that letter | BACKSPACE discard the last burst | ESC save and quit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .data import Samples
from .live import draw_hand
from .tracker import DEFAULT_MODEL_PATH, HandTracker


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--burst", type=int, default=60, help="frames recorded per key press")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args(argv)

    bursts: list[list[tuple]] = []  # each burst is a list of (landmarks, handed, label, source, group)
    session = time.strftime("%Y%m%d-%H%M%S")
    label, remaining = None, 0
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    t0 = time.time()
    try:
        with HandTracker(args.model, video=True) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                hand = tracker.detect(frame, int((time.time() - t0) * 1000))
                if hand is not None:
                    draw_hand(frame, hand.landmarks)
                    if remaining > 0:
                        i = len(bursts) - 1
                        bursts[-1].append((hand.landmarks, hand.handed, label,
                                           f"{session}/burst{i}/{args.burst - remaining}",
                                           f"{args.signer}/{label}/{session}-{i}"))
                        remaining -= 1
                view = cv2.flip(frame, 1)  # mirror for display only; detection used the true image
                total = sum(len(b) for b in bursts)
                status = f"REC {label}: {remaining} left" if remaining else "press a letter to record"
                cv2.putText(view, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255) if remaining else (255, 255, 255), 2)
                cv2.putText(view, f"{total} samples, {len(bursts)} bursts", (12, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("collect", view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if remaining == 0 and key in (8, 127) and bursts:
                    bursts.pop()
                elif remaining == 0 and ord("a") <= key <= ord("z"):
                    label, remaining = chr(key).upper(), args.burst
                    bursts.append([])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rows = [r for b in bursts for r in b]
        if rows:
            new = Samples(landmarks=[r[0] for r in rows], handed=[r[1] for r in rows],
                          labels=[r[2] for r in rows], signer=[args.signer] * len(rows),
                          source=[r[3] for r in rows], group=[r[4] for r in rows])
            if args.out.exists():  # append across sessions
                new = Samples.concat([Samples.load(args.out), new])
            new.save(args.out)
            print(f"[collect] {args.out} now holds {len(new)} samples")


if __name__ == "__main__":
    main()
