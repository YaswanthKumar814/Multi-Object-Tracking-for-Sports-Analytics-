"""Write a short synthetic clip for checking that video I/O works on your machine.

    python scripts/make_synthetic_video.py --out outputs/synthetic.mp4

Note: COCO YOLO will NOT detect the coloured rectangles as people; this clip
is for checking decoding/encoding and the CLI wiring, not detection quality.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_tracker.synthetic import generate_scene, write_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/synthetic.mp4")
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()
    path = write_video(generate_scene(num_frames=args.frames), args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
