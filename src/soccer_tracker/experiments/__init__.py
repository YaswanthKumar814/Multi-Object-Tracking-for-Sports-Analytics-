"""Reproducible experiments: detection caching, tracker comparison, ball evaluation."""

from .detections import DetectionCache, cache_path, compute_detections, get_detections
from .manifest import Manifest, load_manifest

__all__ = [
    "DetectionCache",
    "Manifest",
    "cache_path",
    "compute_detections",
    "get_detections",
    "load_manifest",
]
