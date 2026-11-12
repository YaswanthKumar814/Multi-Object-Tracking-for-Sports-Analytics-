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
from typing import Optional, Sequence

import numpy as np

from ..config import BallTrackerConfig
from ..detection.base import Detections
from .flight import Flight
from .types import BALL, DETECTED, PREDICTED, TrackedObject

MOTION_CLASS = "ball (motion)"


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
        self._last_innovation = 0.0  # px between prediction and the last accepted detection
        self._size = np.array([10.0, 10.0])  # last observed (w, h), used for predicted boxes
        self._class_name = "ball"
        self._last_conf: Optional[float] = None
        self._next_id = 0
        self.track_id: Optional[int] = None
        self.trajectory: deque[TrajectoryPoint] = deque(maxlen=cfg.trajectory_length)
        # Positions where the detector *saw* the ball (not motion-inferred ones),
        # kept across resets: a flight must start at one of them, so a wrong
        # flight can never become the anchor for the next one.
        self._anchor_history: deque[tuple[int, float, float]] = deque(maxlen=60)
        self._flight_id: Optional[int] = None  # flight currently being followed
        # Corrected outputs for earlier frames (frame_idx -> ball), filled when a
        # flight is adopted; the pipeline applies them to frames it still holds.
        self._amendments: dict[int, TrackedObject] = {}
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
        self._flight_id = None
        self.trajectory.clear()

    def take_amendments(self) -> dict[int, TrackedObject]:
        out, self._amendments = self._amendments, {}
        return out

    def _start(self, dets: Detections, i: int) -> None:
        c = dets.centers[i]
        self._kf = ConstantVelocityKalman(c[0], c[1], self.cfg.process_noise, self.cfg.measurement_noise)
        self._hits = 1
        self._missed = 0
        self._last_innovation = 0.0
        self.track_id = self._next_id
        self._next_id += 1
        self.tracks_started += 1
        self.trajectory.clear()
        self._absorb(dets, i)

    def _best_starter(self, dets: Detections) -> Optional[int]:
        """Most confident candidate that is strong enough to start a track, if any.

        Weak candidates may still extend an existing track inside the gate, but
        they must not seed a new one: on broadcast footage the low-confidence
        "sports ball" hits are mostly heads, shoes and pitch markings.
        """
        if not len(dets):
            return None
        i = int(np.argmax(dets.confidence))
        return i if dets.confidence[i] >= self.cfg.start_confidence else None

    def _absorb(self, dets: Detections, i: int) -> None:
        self._flight_id = None  # the detector sees the ball again: the flight is over
        self._size = np.array([dets.widths[i], dets.heights[i]], dtype=float)
        self._class_name = str(dets.class_name[i])
        self._last_conf = float(dets.confidence[i])

    def _box(self, center: np.ndarray) -> np.ndarray:
        w, h = self._size
        return np.array([center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2], dtype=np.float32)

    # -- main entry point --------------------------------------------------- #
    def update(
        self,
        frame_idx: int,
        candidates: Detections,
        frame_shape: Optional[tuple[int, ...]] = None,
        flights: Sequence[Flight] = (),
    ) -> Optional[TrackedObject]:
        """Process one frame. Returns the ball if a confirmed track exists, else None.

        ``flights`` are confirmed motion-based flights matched in this frame
        (tracking/flight.py). An adopted flight overrides the appearance
        detections: while the ball is in flight the detector mostly sees
        heads and boots, and the flight is far stronger evidence.
        """
        flight = self._pick_flight(frame_idx, flights)
        # A confident detection stops a *new* flight from being adopted (a boot
        # in the run-up must not steal a ball the detector can see). A flight
        # already being followed is stronger evidence than a single detection,
        # which during a shot is often the keeper's glove or a boot.
        if flight is not None and flight.id != self._flight_id and self._sees_ball(candidates):
            flight = None
        if flight is not None:
            return self._follow_flight(frame_idx, flight)
        if self._flight_id is not None:
            # The flight we followed was not matched this frame: the ball hit
            # the net, was caught or bounced. Its velocity no longer holds, so
            # do not extrapolate it; only a detection can bring the ball back.
            self._last_innovation = float("inf")
        return self._update_appearance(frame_idx, candidates, frame_shape)

    # -- motion flights ------------------------------------------------------ #
    def _sees_ball(self, candidates: Detections) -> bool:
        """A confident appearance detection inside the gate: trust it over any flight."""
        if self._kf is None or not len(candidates):
            return False
        cfg = self.cfg
        pred = (ConstantVelocityKalman.F @ self._kf.x)[:2]
        gate = min(cfg.gate_px + cfg.gate_growth_px * self._missed, max(cfg.max_gate_px, cfg.gate_px))
        near = np.linalg.norm(candidates.centers - pred[None, :], axis=1) <= gate
        return bool(np.any(near & (candidates.confidence >= cfg.start_confidence)))

    def _last_seen(self, frame_idx: int) -> Optional[tuple[int, np.ndarray]]:
        """(frame, position) where the detector last saw the ball, at or before ``frame_idx``."""
        for f, x, y in reversed(self._anchor_history):
            if f <= frame_idx:
                return (f, np.array([x, y])) if frame_idx - f <= self.cfg.flight_max_anchor_gap else None
        return None

    def _launched_from_ball(self, fl: Flight) -> Optional[bool]:
        """Does the flight come from where the ball was last seen? None = nothing to check against.

        The flight's first velocity, run backwards to the frame the ball was
        last seen, must land on the ball: a kicked ball comes *from* the ball,
        whereas a player or a boot that merely starts nearby does not.
        """
        seen = self._last_seen(fl.first_frame)
        if seen is None:
            return None
        f_seen, anchor = seen
        (t0, x0, y0), (t1, x1, y1) = fl.points[0], fl.points[1]
        v0 = np.array([x1 - x0, y1 - y0]) / max(1, t1 - t0)
        origin = np.array([x0, y0]) - v0 * (t0 - f_seen)
        return bool(np.linalg.norm(origin - anchor) <= self.cfg.flight_anchor_px)

    def _pick_flight(self, frame_idx: int, flights: Sequence[Flight]) -> Optional[Flight]:
        cfg = self.cfg
        for fl in flights:
            if fl.id == self._flight_id:
                return fl  # keep following the flight we adopted
        if self._flight_id is not None and self._kf is not None:
            # Chains can share blobs, so the same flight may continue under a
            # duplicate chain: take one that is where the filter expects the ball.
            expected = (ConstantVelocityKalman.F @ self._kf.x)[:2]
            close = [fl for fl in flights if np.linalg.norm(fl.position - expected) <= cfg.flight_continue_px]
            if close:
                fl = max(close, key=len)
                self._flight_id = fl.id
                return fl
        best = None
        for fl in flights:
            launched = self._launched_from_ball(fl)
            if launched is not None:
                ok = launched
            else:
                # Nothing to anchor to: only when no ball is currently tracked,
                # and only a long flight.
                ok = (
                    cfg.flight_unanchored_min_length > 0
                    and (not self.confirmed or self._missed > 0)
                    and len(fl) >= cfg.flight_unanchored_min_length
                )
            if ok and (best is None or len(fl) > len(best)):
                best = fl
        return best

    def _follow_flight(self, frame_idx: int, fl: Flight) -> Optional[TrackedObject]:
        cfg = self.cfg
        x, y = fl.position
        if fl.id != self._flight_id or self._kf is None:
            # Adopt: restart the filter on the flight with the flight's velocity.
            self._kf = ConstantVelocityKalman(x, y, cfg.process_noise, cfg.measurement_noise)
            self._kf.x[2:] = fl.velocity
            if self.track_id is None:
                self.track_id = self._next_id
                self._next_id += 1
                self.tracks_started += 1
            self._hits = max(self._hits, cfg.min_hits)
            self._flight_id = fl.id
            # Fill in the flight's frames from before it was confirmed.
            for f, px, py in fl.points[:-1]:
                self._amendments[f] = self._object(np.array([px, py]), DETECTED, MOTION_CLASS, None)
        else:
            self._kf.predict()
            self._kf.update(np.array([x, y]))
            # A one-frame gap in the flight: interpolate between the two observations.
            (f1, x1, y1), (f0, x0, y0) = fl.points[-1], fl.points[-2]
            for f in range(f0 + 1, f1):
                a = (f - f0) / (f1 - f0)
                pos = np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0)])
                self._amendments[f] = self._object(pos, PREDICTED, MOTION_CLASS, None)
        self._missed = 0
        self._last_innovation = 0.0  # the flight's velocity is well measured
        self._class_name = MOTION_CLASS
        self._last_conf = None  # no detector score for a motion measurement
        return self._emit(frame_idx, DETECTED)

    # -- appearance detections ---------------------------------------------- #
    def _update_appearance(
        self, frame_idx: int, candidates: Detections, frame_shape: Optional[tuple[int, ...]]
    ) -> Optional[TrackedObject]:
        cfg = self.cfg
        n = len(candidates)

        if self._kf is None:
            best = self._best_starter(candidates)
            if best is not None:
                self._start(candidates, best)
            return self._emit(frame_idx, DETECTED)

        pred = self._kf.predict()
        gate = min(cfg.gate_px + cfg.gate_growth_px * self._missed, max(cfg.max_gate_px, cfg.gate_px))

        if n:
            dist = np.linalg.norm(candidates.centers - pred[None, :], axis=1)
            in_gate = dist <= gate
            if in_gate.any():
                # Prefer confident detections close to the prediction.
                score = np.where(in_gate, candidates.confidence * (1.0 - dist / gate), -np.inf)
                i = int(np.argmax(score))
                self._last_innovation = float(dist[i])
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
            best = self._best_starter(candidates)
            if best is not None:  # try a fresh start from the best remaining candidate
                self._start(candidates, best)
                return self._emit(frame_idx, DETECTED)
            return None

        if self._last_innovation > cfg.max_coast_innovation_px:
            return None  # velocity just changed abruptly; an extrapolation would be a guess
        return self._emit(frame_idx, PREDICTED)

    def _emit(self, frame_idx: int, status: str) -> Optional[TrackedObject]:
        if self._kf is None:
            return None
        pos = self._kf.position
        self.trajectory.append(TrajectoryPoint(frame_idx, float(pos[0]), float(pos[1]), status))
        if status == DETECTED and self.confirmed and self._class_name != MOTION_CLASS:
            self._anchor_history.append((frame_idx, float(pos[0]), float(pos[1])))
        if not self.confirmed:
            return None
        return self._object(pos, status, self._class_name, self._last_conf if status == DETECTED else None)

    def _object(self, pos: np.ndarray, status: str, class_name: str, conf: Optional[float]) -> TrackedObject:
        return TrackedObject(
            object_type=BALL,
            track_id=int(self.track_id),
            xyxy=self._box(pos),
            class_name=class_name,
            confidence=conf,
            status=status,
        )
