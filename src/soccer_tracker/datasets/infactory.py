"""InFactory Soccer Ball Tracking adapter (huggingface.co/datasets/infactory-ai/ball-tracking).

Layout on disk (verified on the downloaded files)::

    <root>/metadata.csv                         file_path, video_source, frame_index, visibility, bboxes_count
    <root>/data/visible/<uuid>_<frame>.jpg      frame with a visible ball
    <root>/data/visible/<uuid>_<frame>.txt      YOLO label: class cx cy w h (normalised to [0, 1])
    <root>/data/not_visible/<uuid>_<frame>.jpg  frame without a visible ball (hard negative)

``metadata.csv`` defines the published sample: 1,450 frames from 10 clips
(884 visible, 566 not visible). Inside each clip the frame indices are
contiguous, so a clip can be replayed in order as a short video.

The labels are ball boxes only. There are no player annotations and no
persistent ball identity, so this dataset is used for **ball detection
evaluation only**; identity-based tracking metrics are never reported for it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .base import DatasetError, Sequence, SequenceInfo

ASSUMED_FPS = 25.0  # the dataset does not publish a frame rate; only used for timestamps


def read_metadata(root: str | Path) -> dict[str, list[dict]]:
    """Clip uuid -> rows sorted by frame index."""
    path = Path(root) / "metadata.csv"
    if not path.is_file():
        raise DatasetError(f"InFactory metadata not found: {path}")
    clips: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["frame_index"] = int(row["frame_index"])
            clips.setdefault(row["video_source"], []).append(row)
    for rows in clips.values():
        rows.sort(key=lambda r: r["frame_index"])
    return clips


def list_clips(root: str | Path) -> list[str]:
    return sorted(read_metadata(root))


class InFactoryClip(Sequence):
    has_ball_boxes = True

    def __init__(self, root: str | Path, clip: str, role: str = "test", max_frames: Optional[int] = None):
        self.root = Path(root)
        clips = read_metadata(self.root)
        if clip not in clips:
            raise DatasetError(f"Clip {clip!r} is not listed in {self.root / 'metadata.csv'}")
        self.rows = clips[clip][:max_frames] if max_frames else clips[clip]
        first = cv2.imread(str(self.root / self.rows[0]["file_path"]))
        if first is None:
            raise DatasetError(f"Could not read {self.root / self.rows[0]['file_path']}")
        h, w = first.shape[:2]
        self.info = SequenceInfo(
            dataset="InFactory", name=clip[:8], split="public sample", role=role, fps=ASSUMED_FPS,
            width=w, height=h, num_frames=len(self.rows),
            source_first_frame=self.rows[0]["frame_index"], source=str(self.root / "data"),
        )
        self.clip = clip

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for i, row in enumerate(self.rows, start=1):
            img = cv2.imread(str(self.root / row["file_path"]))
            if img is None:
                raise DatasetError(f"Could not read {self.root / row['file_path']}")
            yield i, img

    def ball_gt(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for i, row in enumerate(self.rows, start=1):
            if row["visibility"] != "visible":
                continue
            label = (self.root / row["file_path"]).with_suffix(".txt")
            if not label.is_file():
                raise DatasetError(f"Missing label file {label}")
            boxes = parse_yolo_label(label.read_text(encoding="utf-8"), self.info.width, self.info.height)
            if len(boxes):
                out[i] = boxes
        return out


def parse_yolo_label(text: str, width: int, height: int) -> np.ndarray:
    """YOLO ``class cx cy w h`` (normalised) -> ``(K, 4)`` pixel xyxy boxes."""
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
