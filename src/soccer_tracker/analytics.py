"""Per-frame CSV export and a run summary.

Rows are streamed to disk as they are produced, so memory use does not grow
with video length.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .tracking.types import BALL, DETECTED, PREDICTED, TrackedObject

log = logging.getLogger(__name__)

CSV_COLUMNS = [
    "frame",
    "timestamp_s",
    "object_type",  # player | ball
    "track_id",  # player: ByteTrack id; ball: ball-track segment id (increments on re-acquisition)
    "class_name",  # class reported by the detector (e.g. person, sports ball)
    "confidence",  # empty for predicted positions
    "status",  # detected | predicted
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
]


class TrackCsvWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None
        self._writer = None
        self.rows_written = 0

    def open(self) -> "TrackCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_COLUMNS)
        return self

    def write_frame(self, frame_idx: int, timestamp_s: float, objects: Iterable[TrackedObject]) -> None:
        if self._writer is None:
            raise RuntimeError("TrackCsvWriter.write_frame() called before open()")
        for o in objects:
            x1, y1, x2, y2 = (float(v) for v in o.xyxy)
            cx, cy = o.center
            self._writer.writerow(
                [
                    frame_idx,
                    f"{timestamp_s:.4f}",
                    o.object_type,
                    o.track_id,
                    o.class_name,
                    "" if o.confidence is None else f"{o.confidence:.4f}",
                    o.status,
                    f"{x1:.1f}",
                    f"{y1:.1f}",
                    f"{x2:.1f}",
                    f"{y2:.1f}",
                    f"{cx:.1f}",
                    f"{cy:.1f}",
                ]
            )
            self.rows_written += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    def __enter__(self) -> "TrackCsvWriter":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class RunStats:
    frames: int = 0
    player_detections_raw: int = 0
    player_detections_kept: int = 0
    ball_candidates_raw: int = 0
    ball_candidates_kept: int = 0
    ball_frames_detected: int = 0
    ball_frames_predicted: int = 0
    ball_tracks_started: int = 0
    player_track_frames: Counter = field(default_factory=Counter)
    elapsed_s: float = 0.0

    def record(self, players: list[TrackedObject], ball: Optional[TrackedObject]) -> None:
        self.frames += 1
        for p in players:
            self.player_track_frames[p.track_id] += 1
        if ball is not None and ball.object_type == BALL:
            if ball.status == DETECTED:
                self.ball_frames_detected += 1
            elif ball.status == PREDICTED:
                self.ball_frames_predicted += 1

    def to_dict(self, extra: Optional[dict] = None) -> dict:
        lengths = list(self.player_track_frames.values())
        d = {
            "frames_processed": self.frames,
            "elapsed_s": round(self.elapsed_s, 2),
            "processing_fps": round(self.frames / self.elapsed_s, 2) if self.elapsed_s > 0 else None,
            "players": {
                "detections_raw": self.player_detections_raw,
                "detections_after_filter": self.player_detections_kept,
                "unique_track_ids": len(lengths),
                "median_track_length_frames": statistics.median(lengths) if lengths else 0,
                # Many very short tracks usually means fragmentation / ID switches.
                "tracks_shorter_than_10_frames": sum(1 for n in lengths if n < 10),
            },
            "ball": {
                "candidates_raw": self.ball_candidates_raw,
                "candidates_after_filter": self.ball_candidates_kept,
                "frames_detected": self.ball_frames_detected,
                "frames_predicted": self.ball_frames_predicted,
                "frames_lost": self.frames - self.ball_frames_detected - self.ball_frames_predicted,
                "coverage": round((self.ball_frames_detected + self.ball_frames_predicted) / self.frames, 3)
                if self.frames
                else 0.0,
                "tracks_started": self.ball_tracks_started,
            },
        }
        if extra:
            d.update(extra)
        return d


def write_summary(path: str | Path, summary: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
