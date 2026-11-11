"""Turn a folder of numbered frames (e.g. a SportsMOT sequence) into an MP4 the application can read.

    python scripts/frames_to_video.py data/raw/sportsmot/val/v_dw7LOz17Omg_c067/img1 results/demo/input_sportsmot.mp4 --fps 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("output")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    files = sorted(p for p in Path(args.frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.max_frames:
        files = files[: args.max_frames]
    if not files:
        print(f"No images found in {args.frames_dir}", file=sys.stderr)
        return 2
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        print(f"Could not open a video writer for {out}", file=sys.stderr)
        return 2
    try:
        for p in files:
            img = cv2.imread(str(p))
            if img is None or img.shape[:2] != (h, w):
                print(f"Skipping unreadable or differently sized frame {p}", file=sys.stderr)
                continue
            writer.write(img)
    finally:
        writer.release()
    print(f"Wrote {len(files)} frames ({w}x{h} @ {args.fps} FPS) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
