from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

PLAYER = "player"
BALL = "ball"

DETECTED = "detected"  # position comes from a detection in this frame
PREDICTED = "predicted"  # position is a motion-model estimate (no detection matched)


@dataclass
class TrackedObject:
    object_type: str  # PLAYER or BALL
    track_id: int
    xyxy: np.ndarray  # (4,) float
    class_name: str
    confidence: Optional[float]  # None for predicted positions
    status: str = DETECTED

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return float((x1 + x2) / 2), float(y2)
