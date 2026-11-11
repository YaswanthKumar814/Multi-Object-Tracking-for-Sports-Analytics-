"""Common interface for labelled evaluation sequences.

Each dataset has its own adapter (one module per dataset) because the three
datasets store frames and annotations differently. Adapters convert to one
small in-memory representation so the experiment code never touches
dataset-specific files:

* frames are numbered 1..N *within the evaluated range* (MOT convention);
* player ground truth is an ``(M, 6)`` array ``[frame, track_id, x, y, w, h]``
  with top-left ``x, y`` in pixels;
* ball ground truth maps a frame number to a ``(K, 4)`` array of ``xyxy`` boxes.

Annotations from different datasets are never mixed: every sequence carries the
name of the dataset it came from, and results are always reported per dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


class DatasetError(Exception):
    """A dataset directory is missing or does not have the expected layout."""


@dataclass(frozen=True)
class SequenceInfo:
    dataset: str  # "SportsMOT", "UVY-Track", "InFactory"
    name: str
    split: str  # split as published by the dataset authors
    role: str  # how this project uses it: "dev" (tuning) or "test"
    fps: float
    width: int
    height: int
    num_frames: int  # frames in the evaluated range
    source_first_frame: int  # frame number in the original sequence that maps to frame 1
    source: str  # path of the frames / video

    @property
    def key(self) -> str:
        """Filesystem-safe unique identifier."""
        return f"{self.dataset}__{self.name}".replace("-", "_").replace(" ", "_")


class Sequence:
    """Base class; adapters override what their dataset provides."""

    info: SequenceInfo
    has_player_ids: bool = False  # persistent player identities -> tracking metrics allowed
    has_ball_boxes: bool = False  # ball bounding boxes -> ball detection metrics allowed
    has_ball_ids: bool = False  # persistent ball identity

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_number, bgr_image)`` for frame_number = 1..num_frames."""
        raise NotImplementedError

    def player_gt(self) -> np.ndarray:
        """``(M, 6)`` float array ``[frame, track_id, x, y, w, h]``; empty if unavailable."""
        return np.zeros((0, 6), dtype=np.float64)

    def ball_gt(self) -> dict[int, np.ndarray]:
        """Frames in which the ball is annotated as visible -> ``(K, 4)`` xyxy boxes."""
        return {}

    def ball_labelled_frames(self) -> Optional[set[int]]:
        """Frames whose ball status (visible or not) is known. ``None`` = every frame."""
        return None

    def describe(self) -> dict:
        gt = self.player_gt()
        balls = self.ball_gt()
        return {
            "dataset": self.info.dataset,
            "sequence": self.info.name,
            "split": self.info.split,
            "role": self.info.role,
            "fps": self.info.fps,
            "width": self.info.width,
            "height": self.info.height,
            "frames_evaluated": self.info.num_frames,
            "source_first_frame": self.info.source_first_frame,
            "source_last_frame": self.info.source_first_frame + self.info.num_frames - 1,
            "player_gt_boxes": int(len(gt)),
            "player_gt_tracks": int(len(np.unique(gt[:, 1]))) if len(gt) else 0,
            "ball_visible_frames": int(len(balls)),
            "has_player_ids": self.has_player_ids,
            "has_ball_boxes": self.has_ball_boxes,
            "has_ball_ids": self.has_ball_ids,
        }
