"""UVY-Track adapter (Alencar & de Freitas; Zenodo record 21303900).

Layout on disk (verified on the downloaded files)::

    <root>/<sequence>/<video>.mp4      the source clip; frame i of the video is img1/00000i.jpg
    <root>/<sequence>/video_info.txt   key=value lines (FPS, numFrames, imWidth, imHeight, ...)
    <root>/<sequence>/gt/labels.txt    one class name per line
    <root>/<sequence>/gt/gt.txt        frame, id, x, y, w, h, conf, class, visibility

This is **not** YOLO format. ``gt.txt`` is MOT-style with pixel ``x, y, w, h``
(top-left corner) and a persistent track id per object; ``class`` is a 1-based
index into ``labels.txt`` (1 player, 2 sports ball, 3 goal, 4 referee,
5 goalkeeper, 6 pcamera).

Notes from inspecting the data:

* Several clips are annotated only for part of their length, so an explicit
  frame range is evaluated and frames outside it are never scored.
* The detector used in this project reports the generic COCO ``person`` class,
  so the three person classes (player, goalkeeper, referee) are merged into one
  class-agnostic "person" target for player tracking. ``goal`` and ``pcamera``
  are ignored.
* Rows with a negative track id are annotation leftovers and are dropped.
* ``video_info.txt`` can be wrong (soccer_V01 names a different sequence), so
  the image size and frame rate are read from the video itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .base import DatasetError, Sequence, SequenceInfo

PERSON_CLASSES = ("player", "goalkeeper", "referee")
BALL_CLASS = "sports ball"


class UVYTrackSequence(Sequence):
    has_player_ids = True
    has_ball_boxes = True
    has_ball_ids = True

    def __init__(self, root: str | Path, name: str, first_frame: Optional[int] = None,
                 last_frame: Optional[int] = None, role: str = "test"):
        self.dir = Path(root) / name
        videos = sorted(self.dir.glob("*.mp4"))
        gt_path = self.dir / "gt" / "gt.txt"
        labels_path = self.dir / "gt" / "labels.txt"
        if not videos or not gt_path.is_file() or not labels_path.is_file():
            raise DatasetError(f"UVY-Track sequence not found or incomplete: {self.dir}")
        self.video = videos[0]
        self.labels = [ln.strip() for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

        rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
        rows = rows[(rows[:, 1] >= 0) & (rows[:, 6] != 0)]
        annotated = rows[:, 0].astype(int)
        self.first = int(first_frame) if first_frame else int(annotated.min())
        self.last = int(last_frame) if last_frame else int(annotated.max())
        if self.last < self.first:
            raise DatasetError(f"Empty frame range {self.first}..{self.last} for {name}")
        keep = (annotated >= self.first) & (annotated <= self.last)
        self._rows = rows[keep]

        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise DatasetError(f"Could not open video {self.video}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        self.info = SequenceInfo(
            dataset="UVY-Track", name=name, split="full (no official split)", role=role, fps=float(fps),
            width=width, height=height, num_frames=self.last - self.first + 1,
            source_first_frame=self.first, source=str(self.video),
        )

    def _class_ids(self, names: tuple[str, ...]) -> list[int]:
        return [i + 1 for i, n in enumerate(self.labels) if n in names]

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.video))
        try:
            idx = 0  # 1-based number of the frame just decoded
            while idx < self.last:
                ok, frame = cap.read()
                if not ok:
                    raise DatasetError(f"{self.video} ended at frame {idx}, before frame {self.last}")
                idx += 1
                if idx >= self.first:
                    yield idx - self.first + 1, frame
        finally:
            cap.release()

    def player_gt(self) -> np.ndarray:
        r = self._rows[np.isin(self._rows[:, 7].astype(int), self._class_ids(PERSON_CLASSES))]
        out = r[:, :6].astype(np.float64).copy()
        out[:, 0] -= self.first - 1
        return out

    def ball_gt(self) -> dict[int, np.ndarray]:
        r = self._rows[np.isin(self._rows[:, 7].astype(int), self._class_ids((BALL_CLASS,)))]
        out: dict[int, list] = {}
        for row in r:
            x, y, w, h = row[2:6]
            out.setdefault(int(row[0]) - self.first + 1, []).append([x, y, x + w, y + h])
        return {f: np.asarray(b, dtype=np.float64) for f, b in out.items()}

    def ball_track_ids(self) -> dict[int, int]:
        """Frame -> ground-truth ball track id (the dataset labels one ball per frame)."""
        r = self._rows[np.isin(self._rows[:, 7].astype(int), self._class_ids((BALL_CLASS,)))]
        return {int(row[0]) - self.first + 1: int(row[1]) for row in r}

    def ball_labelled_frames(self) -> Optional[set[int]]:
        # A frame is trusted as "ball not visible" only if something else was
        # annotated in it; frames with no rows at all may simply be unlabelled.
        return {int(f) - self.first + 1 for f in np.unique(self._rows[:, 0])}
