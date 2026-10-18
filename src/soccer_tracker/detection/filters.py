"""Cheap geometric / colour filters applied after detection.

These remove obvious false positives (crowd members, tiny blobs, a "ball"
the size of a player) before they reach the trackers.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import BallFilterConfig, PitchMaskConfig, PlayerFilterConfig
from .base import Detections


class PitchMasker:
    """Binary mask of grass-coloured pixels (HSV threshold + closing + dilation)."""

    def __init__(self, cfg: PitchMaskConfig):
        self.lower = np.array(cfg.hsv_lower, dtype=np.uint8)
        self.upper = np.array(cfg.hsv_upper, dtype=np.uint8)
        self.dilate_px = cfg.dilate_px

    def compute(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # Close holes left by players/lines, then grow so feet at the edge still count.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        if self.dilate_px > 0:
            k = 2 * self.dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask > 0


def _points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xs = np.clip(points[:, 0].astype(int), 0, w - 1)
    ys = np.clip(points[:, 1].astype(int), 0, h - 1)
    return mask[ys, xs]


def filter_players(
    dets: Detections, frame_shape: tuple[int, ...], cfg: PlayerFilterConfig, pitch_mask: Optional[np.ndarray] = None
) -> Detections:
    if len(dets) == 0:
        return dets
    h_frame = frame_shape[0]
    heights = dets.heights
    widths = np.maximum(dets.widths, 1e-6)
    aspect = heights / widths
    keep = (
        (heights >= cfg.min_height_px)
        & (heights <= cfg.max_height_frac * h_frame)
        & (aspect >= cfg.min_aspect)
        & (aspect <= cfg.max_aspect)
    )
    if cfg.use_pitch_mask and pitch_mask is not None:
        keep &= _points_in_mask(dets.bottom_centers, pitch_mask)
    return dets.subset(keep)


def filter_balls(dets: Detections, cfg: BallFilterConfig, pitch_mask: Optional[np.ndarray] = None) -> Detections:
    if len(dets) == 0:
        return dets
    w = dets.widths
    h = dets.heights
    side = np.maximum(w, h)
    ratio = side / np.maximum(np.minimum(w, h), 1e-6)
    keep = (side >= cfg.min_size_px) & (side <= cfg.max_size_px) & (ratio <= cfg.max_aspect_ratio)
    if cfg.use_pitch_mask and pitch_mask is not None:
        keep &= _points_in_mask(dets.centers, pitch_mask)
    return dets.subset(keep)
