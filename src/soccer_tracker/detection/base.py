"""Detector-agnostic detection container and helpers.

Nothing in this module imports a deep-learning framework, so trackers,
filters, and tests can use it without PyTorch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Detections:
    """Axis-aligned boxes for one frame.

    xyxy:        (N, 4) float32, pixel coordinates (x1, y1, x2, y2)
    confidence:  (N,) float32
    class_name:  (N,) object array of str
    """

    xyxy: np.ndarray
    confidence: np.ndarray
    class_name: np.ndarray

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self.class_name = np.asarray(self.class_name, dtype=object).reshape(-1)
        n = len(self.xyxy)
        if len(self.confidence) != n or len(self.class_name) != n:
            raise ValueError(
                f"Detections fields disagree in length: xyxy={n}, "
                f"confidence={len(self.confidence)}, class_name={len(self.class_name)}"
            )

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=object))

    @classmethod
    def concat(cls, items: Sequence["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return cls.empty()
        return cls(
            np.concatenate([d.xyxy for d in items]),
            np.concatenate([d.confidence for d in items]),
            np.concatenate([d.class_name for d in items]),
        )

    def __len__(self) -> int:
        return len(self.xyxy)

    def subset(self, mask_or_idx: np.ndarray) -> "Detections":
        return Detections(self.xyxy[mask_or_idx], self.confidence[mask_or_idx], self.class_name[mask_or_idx])

    @property
    def widths(self) -> np.ndarray:
        return self.xyxy[:, 2] - self.xyxy[:, 0]

    @property
    def heights(self) -> np.ndarray:
        return self.xyxy[:, 3] - self.xyxy[:, 1]

    @property
    def centers(self) -> np.ndarray:
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2], axis=1)

    @property
    def bottom_centers(self) -> np.ndarray:
        """Foot point of a player box; the most stable point for ground trajectories."""
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]], axis=1)


class Detector(Protocol):
    """Anything that turns a BGR frame into ``Detections``."""

    def detect(self, frame: np.ndarray) -> Detections: ...


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N, M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def nms(dets: Detections, iou_threshold: float) -> Detections:
    """Class-aware greedy non-maximum suppression."""
    if len(dets) <= 1:
        return dets
    keep: list[int] = []
    for cls in np.unique(dets.class_name):
        idx = np.flatnonzero(dets.class_name == cls)
        order = idx[np.argsort(-dets.confidence[idx], kind="stable")]
        while len(order):
            best = order[0]
            keep.append(int(best))
            if len(order) == 1:
                break
            ious = box_iou(dets.xyxy[best][None], dets.xyxy[order[1:]])[0]
            order = order[1:][ious <= iou_threshold]
    keep.sort()
    return dets.subset(np.array(keep, dtype=int))


def compute_tiles(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tile rectangles (x1, y1, x2, y2) covering the whole frame.

    The last tile in each row/column is shifted inwards so every tile has the
    full size whenever the frame is large enough.
    """
    step = max(1, int(round(tile * (1.0 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        s = list(range(0, length - tile, step))
        s.append(length - tile)
        return sorted(set(s))

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]
