"""Single-target ball tracker.

Why a dedicated tracker instead of ByteTrack for the ball?
* There is at most one match ball, so global assignment is unnecessary;
  the hard problem is choosing *which* candidate (if any) is the ball.
* Ball detections are sparse and noisy. A constant-velocity Kalman filter
  lets us (a) reject candidates far from where the ball should be,
  (b) coast through short gaps (occlusion, motion blur) and mark those
  positions as "predicted", and (c) re-acquire after a long pass or a
  camera cut when a confident detection appears elsewhere.

Life cycle: NONE -> TENTATIVE (hits < min_hits, not reported)
            -> CONFIRMED (reported) -> NONE after too many misses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import BallTrackerConfig
from ..detection.base import Detections
from .types import BALL, DETECTED, PREDICTED, TrackedObject


class ConstantVelocityKalman:
    """2-D constant-velocity Kalman filter, state = [x, y, vx, vy], dt = 1 frame.

    process_noise:     std-dev of unmodelled acceleration (px / frame^2)
    measurement_noise: std-dev of detection centre error (px)
    """

    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def __init__(self, x: float, y: float, process_noise: float, measurement_noise: float):
        self.x = np.array([x, y, 0.0, 0.0])
        r2 = measurement_noise**2
        self.P = np.diag([r2, r2, 25.0**2, 25.0**2])  # velocity unknown at start
        g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        self.Q = (process_noise**2) * (g @ g.T)
        self.R = np.eye(2) * r2

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z: np.ndarray) -> None:
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()


@dataclass
class TrajectoryPoint:
    frame_idx: int
    x: float
    y: float
    status: str


class BallTracker:
    def __init__(self, cfg: BallTrackerConfig):
        self.cfg = cfg
        self._kf: Optional[ConstantVelocityKalman] = None
        self._hits = 0
        self._missed = 0
        self._size = np.array([10.0, 10.0])  # last observed (w, h), used for predicted boxes
        self._class_name = "ball"
        self._last_conf: Optional[float] = None
        self._next_id = 0
        self.track_id: Optional[int] = None
        self.trajectory: deque[TrajectoryPoint] = deque(maxlen=cfg.trajectory_length)
        # counters for the run summary
        self.tracks_started = 0

    # -- state helpers ----------------------------------------------------- #
    @property
    def active(self) -> bool:
        return self._kf is not None

    @property
    def confirmed(self) -> bool:
        return self.active and self._hits >= self.cfg.min_hits

    def reset(self) -> None:
        self._kf = None
        self._hits = 0
        self._missed = 0
        self.track_id = None
        self.trajectory.clear()

    def _start(self, dets: Detections, i: int) -> None:
        c = dets.centers[i]
        self._kf = ConstantVelocityKalman(c[0], c[1], self.cfg.process_noise, self.cfg.measurement_noise)
        self._hits = 1
        self._missed = 0
        self.track_id = self._next_id
        self._next_id += 1
        self.tracks_started += 1
        self.trajectory.clear()
        self._absorb(dets, i)

    def _absorb(self, dets: Detections, i: int) -> None:
        self._size = np.array([dets.widths[i], dets.heights[i]], dtype=float)
        self._class_name = str(dets.class_name[i])
        self._last_conf = float(dets.confidence[i])

    def _box(self, center: np.ndarray) -> np.ndarray:
        w, h = self._size
        return np.array([center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2], dtype=np.float32)

    # -- main entry point --------------------------------------------------- #
    def update(
        self, frame_idx: int, candidates: Detections, frame_shape: Optional[tuple[int, ...]] = None
    ) -> Optional[TrackedObject]:
        """Process one frame. Returns the ball if a confirmed track exists, else None."""
        cfg = self.cfg
        n = len(candidates)

        if self._kf is None:
            if n:
                self._start(candidates, int(np.argmax(candidates.confidence)))
            return self._emit(frame_idx, DETECTED)

        pred = self._kf.predict()
        gate = cfg.gate_px + cfg.gate_growth_px * self._missed

        if n:
            dist = np.linalg.norm(candidates.centers - pred[None, :], axis=1)
            in_gate = dist <= gate
            if in_gate.any():
                # Prefer confident detections close to the prediction.
                score = np.where(in_gate, candidates.confidence * (1.0 - dist / gate), -np.inf)
                i = int(np.argmax(score))
                self._kf.update(candidates.centers[i])
                self._absorb(candidates, i)
                self._hits += 1
                self._missed = 0
                return self._emit(frame_idx, DETECTED)

        # No candidate inside the gate.
        self._missed += 1
        strong = np.flatnonzero(candidates.confidence >= cfg.reinit_confidence) if n else np.array([], int)

        if len(strong) and (self._missed >= cfg.reinit_after_missed or not self.confirmed):
            best = int(strong[np.argmax(candidates.confidence[strong])])
            self._start(candidates, best)
            return self._emit(frame_idx, DETECTED)

        out_of_frame = False
        if frame_shape is not None:
            h, w = frame_shape[:2]
            x, y = pred
            margin = gate
            out_of_frame = x < -margin or y < -margin or x > w + margin or y > h + margin

        if not self.confirmed or self._missed > cfg.max_missed_frames or out_of_frame:
            self.reset()
            if n:  # try a fresh start from the best remaining candidate
                self._start(candidates, int(np.argmax(candidates.confidence)))
                return self._emit(frame_idx, DETECTED)
            return None

        return self._emit(frame_idx, PREDICTED)

    def _emit(self, frame_idx: int, status: str) -> Optional[TrackedObject]:
        if self._kf is None:
            return None
        pos = self._kf.position
        self.trajectory.append(TrajectoryPoint(frame_idx, float(pos[0]), float(pos[1]), status))
        if not self.confirmed:
            return None
        return TrackedObject(
            object_type=BALL,
            track_id=int(self.track_id),
            xyxy=self._box(pos),
            class_name=self._class_name,
            confidence=self._last_conf if status == DETECTED else None,
            status=status,
        )
