"""Evaluation: MOT metrics via trackers.eval, plus custom detection and ball evaluators."""

from .ball import BallOutput, evaluate_ball, pool
from .detection import average_precision, detection_counts, match_frame, prf
from .mot import aggregate, evaluate_mot, flatten, write_mot_file

__all__ = [
    "BallOutput",
    "aggregate",
    "average_precision",
    "detection_counts",
    "evaluate_ball",
    "evaluate_mot",
    "flatten",
    "match_frame",
    "pool",
    "prf",
    "write_mot_file",
]
