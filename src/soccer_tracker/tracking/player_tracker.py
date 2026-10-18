"""Multi-object player tracking with ByteTrack.

Uses ``trackers.ByteTrackTracker`` from Roboflow's ``trackers`` package
(Apache-2.0), the maintained successor of ``supervision.ByteTrack``. The
third-party API is confined to this file: if it changes, nothing else in the
project needs to.
"""

from __future__ import annotations

import numpy as np

from ..config import ByteTrackConfig
from ..detection.base import Detections
from ..errors import DependencyError
from .types import DETECTED, PLAYER, TrackedObject


class PlayerTracker:
    def __init__(self, cfg: ByteTrackConfig, frame_rate: float):
        try:
            import supervision as sv
            from trackers import ByteTrackTracker
        except ImportError as exc:
            raise DependencyError(
                "Player tracking needs the 'trackers' and 'supervision' packages. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        self._sv = sv
        self._tracker = ByteTrackTracker(
            lost_track_buffer=cfg.lost_track_buffer,
            frame_rate=float(frame_rate),
            track_activation_threshold=cfg.track_activation_threshold,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            minimum_iou_threshold=cfg.minimum_iou_threshold,
            high_conf_det_threshold=cfg.high_conf_det_threshold,
        )
        self._class_ids: dict[str, int] = {}
        self._class_names: dict[int, str] = {}

    def _class_id(self, name: str) -> int:
        if name not in self._class_ids:
            idx = len(self._class_ids)
            self._class_ids[name] = idx
            self._class_names[idx] = name
        return self._class_ids[name]

    def update(self, dets: Detections) -> list[TrackedObject]:
        """Feed one frame's detections (call every frame, even when empty)."""
        sv = self._sv
        if len(dets):
            sv_dets = sv.Detections(
                xyxy=dets.xyxy.astype(np.float32),
                confidence=dets.confidence.astype(np.float32),
                class_id=np.array([self._class_id(str(c)) for c in dets.class_name], dtype=int),
            )
        else:
            sv_dets = sv.Detections.empty()
        result = self._tracker.update(sv_dets)

        tracked: list[TrackedObject] = []
        if result.tracker_id is None:
            return tracked
        for i, tid in enumerate(result.tracker_id):
            if tid < 0:  # detection not (yet) part of a confirmed track
                continue
            cls_id = int(result.class_id[i]) if result.class_id is not None else 0
            conf = float(result.confidence[i]) if result.confidence is not None else None
            tracked.append(
                TrackedObject(
                    object_type=PLAYER,
                    track_id=int(tid),
                    xyxy=np.asarray(result.xyxy[i], dtype=np.float32),
                    class_name=self._class_names.get(cls_id, "person"),
                    confidence=conf,
                    status=DETECTED,
                )
            )
        return tracked
