"""Link motion blobs into ball *flights*: fast, smooth, near-constant-velocity paths.

Motion candidates (detection/motion.py) are noisy: every moving boot, sock
or shirt number produces some. A single frame cannot tell them apart from a
ball, but a short sequence can. A struck ball travels quickly and smoothly,
almost in a straight line, for many frames. A limb swings back and forth and
a running player moves slowly. So candidates are linked frame to frame under
a constant-velocity model, and a chain counts as a flight only once it is
long, fast and smooth enough.

Each chain keeps every point it has matched, so when it is confirmed the
tracker can fill in the first few frames of the flight retroactively.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import MotionBallConfig
from ..detection.base import Detections


@dataclass
class Flight:
    id: int
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame_idx, x, y)
    misses: int = 0
    confirmed: bool = False

    @property
    def last_frame(self) -> int:
        return self.points[-1][0]

    @property
    def first_frame(self) -> int:
        return self.points[0][0]

    @property
    def position(self) -> np.ndarray:
        return np.array(self.points[-1][1:], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        """Pixels per frame, from the last two matched points."""
        (t1, x1, y1), (t0, x0, y0) = self.points[-1], self.points[-2]
        return np.array([x1 - x0, y1 - y0], dtype=float) / max(1, t1 - t0)

    def __len__(self) -> int:
        return len(self.points)


class FlightLinker:
    """Stateful: call ``update`` once per frame with that frame's motion candidates."""

    def __init__(self, cfg: MotionBallConfig):
        self.cfg = cfg
        self._chains: list[Flight] = []
        self._seeds = np.zeros((0, 2))  # unmatched candidates from the previous frame
        self._seed_frame = -1
        self._next_id = 0

    def reset(self) -> None:
        self._chains.clear()
        self._seeds = np.zeros((0, 2))
        self._seed_frame = -1

    def update(self, frame_idx: int, candidates: Detections) -> list[Flight]:
        """Advance every chain; return the confirmed flights matched in this frame."""
        cfg = self.cfg
        pts = candidates.centers if len(candidates) else np.zeros((0, 2))
        used = np.zeros(len(pts), dtype=bool)
        alive: list[Flight] = []

        # Chains may share a candidate: an exclusive assignment lets a long junk
        # chain with a wide radius steal the ball's blob. Duplicates are harmless
        # (the tracker follows one flight); ``used`` only stops reseeding.
        for ch in self._chains:
            v = ch.velocity
            pred = ch.position + v * (frame_idx - ch.last_frame)
            radius = cfg.link_radius_px + cfg.link_radius_speed_gain * float(np.linalg.norm(v))
            matched = False
            if len(pts):
                d = np.linalg.norm(pts - pred, axis=1)
                i = int(np.argmin(d))
                if d[i] <= radius:
                    ch.points.append((frame_idx, float(pts[i, 0]), float(pts[i, 1])))
                    ch.misses = 0
                    used[i] = True
                    matched = True
            if not matched:
                ch.misses += 1
            if ch.misses <= cfg.link_max_misses:
                if matched:
                    # Re-checked on every frame: a chain that slows down or
                    # starts wobbling (a player, not a ball) stops being a flight.
                    ch.confirmed = self._is_flight(ch)
                alive.append(ch)

        # New two-point chains from last frame's leftovers to this frame's leftovers.
        if len(self._seeds) and len(pts) and self._seed_frame == frame_idx - 1:
            free = np.flatnonzero(~used)
            if len(free):
                d = np.linalg.norm(pts[free, None, :] - self._seeds[None, :, :], axis=2)
                for a, b in zip(*np.nonzero((d >= cfg.link_min_step_px) & (d <= cfg.link_max_step_px))):
                    i = int(free[a])
                    alive.append(
                        Flight(
                            self._next_id,
                            [(frame_idx - 1, float(self._seeds[b, 0]), float(self._seeds[b, 1])),
                             (frame_idx, float(pts[i, 0]), float(pts[i, 1]))],
                        )
                    )
                    self._next_id += 1
        # Bound the work if a busy scene spawns too many pairs: keep the longest.
        if len(alive) > cfg.link_max_chains:
            alive = sorted(alive, key=len, reverse=True)[: cfg.link_max_chains]

        self._chains = alive
        self._seeds = pts[~used]
        self._seed_frame = frame_idx
        return [ch for ch in self._chains if ch.confirmed and ch.last_frame == frame_idx]

    def _is_flight(self, ch: Flight) -> bool:
        cfg = self.cfg
        if len(ch) < cfg.flight_min_length:
            return False
        p = np.array([q[1:] for q in ch.points], dtype=float)
        t = np.array([q[0] for q in ch.points], dtype=float)
        v = np.diff(p, axis=0) / np.diff(t)[:, None]
        speed = np.linalg.norm(v, axis=1)
        if np.median(speed) < cfg.flight_min_speed_px:
            return False
        # Smoothness relative to speed: a limb reverses (change ~ 2x its speed),
        # a ball changes velocity by a small fraction of its speed; blob-centre
        # jitter of a blurred ball grows with speed, hence the relative term.
        if len(v) >= 2:
            accel = float(np.median(np.linalg.norm(np.diff(v, axis=0), axis=1)))
            if accel > max(cfg.flight_max_accel_px, cfg.flight_max_accel_frac * float(np.median(speed))):
                return False
        # Mostly one direction: net displacement close to the path length.
        net = float(np.linalg.norm(p[-1] - p[0]))
        return net >= 0.7 * cfg.flight_min_speed_px * (t[-1] - t[0])
