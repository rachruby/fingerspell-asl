"""Image folder -> landmark dataset.

Expects root/<LABEL>/<image files>, which is how the Kaggle "ASL Alphabet"
set is laid out. Images where no hand is found are logged to a CSV, not
silently dropped, because per-class detection rate is a data-quality result
in its own right.

    python -m fingerspell.extract --root data/raw/asl_alphabet_train --out data/processed/kaggle.npz
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import cv2

from .data import Samples
from .tracker import DEFAULT_MODEL_PATH, HandTracker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def frame_index(path: Path) -> int:
    """'A1234.jpg' -> 1234, so files sort in capture order."""
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--signer", default="kaggle", help="signer id stored with every sample")
    ap.add_argument("--exclude", nargs="*", default=["nothing"], help="class folders to skip")
    ap.add_argument("--limit-per-class", type=int, default=0, help="evenly subsample each class (0 = all)")
    ap.add_argument("--block", type=int, default=100, help="consecutive frames per split group")
    ap.add_argument("--min-confidence", type=float, default=0.3)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args(argv)

    class_dirs = sorted(d for d in args.root.iterdir() if d.is_dir() and d.name not in args.exclude)
    if not class_dirs:
        raise SystemExit(f"no class folders found under {args.root}")

    rows = {k: [] for k in ("landmarks", "handed", "labels", "signer", "source", "group")}
    failures, t0 = [], time.time()
    with HandTracker(args.model, video=False, min_confidence=args.min_confidence) as tracker:
        for d in class_dirs:
            files = sorted((f for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES), key=frame_index)
            if args.limit_per_class and len(files) > args.limit_per_class:
                step = len(files) / args.limit_per_class
                files = [files[int(i * step)] for i in range(args.limit_per_class)]
            found = 0
            for f in files:
                image = cv2.imread(str(f))
                hand = tracker.detect(image) if image is not None else None
                if hand is None:
                    failures.append((d.name, str(f), "unreadable" if image is None else "no_hand"))
                    continue
                found += 1
                rows["landmarks"].append(hand.landmarks)
                rows["handed"].append(hand.handed)
                rows["labels"].append(d.name)
                rows["signer"].append(args.signer)
                rows["source"].append(str(f))
                rows["group"].append(f"{args.signer}/{d.name}/{frame_index(f) // args.block}")
            rate = found / max(len(files), 1)
            print(f"[extract] {d.name:>8}: {found}/{len(files)} hands found ({rate:.1%})  [{time.time() - t0:.0f}s]")

    if not rows["labels"]:
        raise SystemExit("no hands detected in any image")
    Samples(**rows).save(args.out)
    fail_path = args.out.with_suffix(".failures.csv")
    with open(fail_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "source", "reason"])
        writer.writerows(failures)
    print(f"[extract] wrote {len(rows['labels'])} samples to {args.out}; {len(failures)} failures in {fail_path}")


if __name__ == "__main__":
    main()
