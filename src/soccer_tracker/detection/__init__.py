"""Detection: data container, YOLO wrapper, and post-detection filters."""

from .base import Detections, Detector, box_iou, compute_tiles, nms
from .filters import PitchMasker, filter_balls, filter_players

__all__ = [
    "Detections",
    "Detector",
    "PitchMasker",
    "box_iou",
    "compute_tiles",
    "filter_balls",
    "filter_players",
    "nms",
]
