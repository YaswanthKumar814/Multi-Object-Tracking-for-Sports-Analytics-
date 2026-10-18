"""Command-line entry point.

    python -m soccer_tracker --input match.mp4
    python -m soccer_tracker --input match.mp4 --config configs/default.yaml --max-frames 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from .config import load_config, validate
from .errors import SoccerTrackerError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soccer_tracker",
        description="Detect and track soccer players and the ball; write an annotated video and a CSV of tracks.",
    )
    p.add_argument("-i", "--input", required=True, help="Path to the input video file")
    p.add_argument("-c", "--config", default=None, help="YAML config (default: built-in defaults)")
    p.add_argument("-o", "--output-video", default=None, help="Annotated output video path (overrides config)")
    p.add_argument("--output-csv", default=None, help="CSV output path (overrides config)")
    p.add_argument("--output-summary", default=None, help="JSON run-summary path (overrides config)")
    p.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")
    p.add_argument("--player-weights", default=None, help="Override players.detector.weights")
    p.add_argument("--ball-weights", default=None, help="Override ball.detector.weights")
    p.add_argument("--device", default=None, help="Inference device, e.g. cpu, 0 (CUDA), mps")
    p.add_argument("--no-ball", action="store_true", help="Disable ball detection/tracking")
    p.add_argument("--ball-tiling", action="store_true", help="Enable tiled (sliced) inference for the ball")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("soccer_tracker")

    try:
        cfg = load_config(args.config)
        if args.max_frames is not None:
            cfg.io.max_frames = args.max_frames
        if args.player_weights:
            cfg.players.detector.weights = args.player_weights
        if args.ball_weights:
            cfg.ball.detector.weights = args.ball_weights
        if args.device is not None:
            cfg.players.detector.device = args.device
            cfg.ball.detector.device = args.device
        if args.no_ball:
            cfg.ball.enabled = False
        if args.ball_tiling:
            cfg.ball.detector.tiling.enabled = True
        validate(cfg)

        from .pipeline import Pipeline  # imported here so --help stays fast

        Pipeline(cfg).run(args.input, args.output_video, args.output_csv, args.output_summary)
        return 0
    except SoccerTrackerError as exc:
        log.error("%s", exc)
        if args.verbose and exc.__cause__ is not None:
            log.exception("Underlying error:", exc_info=exc.__cause__)
        return 2
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
