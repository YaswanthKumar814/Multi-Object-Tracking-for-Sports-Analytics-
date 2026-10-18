"""Drawing boxes, IDs, trajectories, and a status overlay onto frames."""

from __future__ import annotations

import colorsys
from collections import deque
from typing import Iterable, Optional

import cv2
import numpy as np

from .config import VisualizationConfig
from .tracking.ball_tracker import TrajectoryPoint
from .tracking.types import PREDICTED, TrackedObject

BALL_COLOR = (0, 255, 255)  # yellow (BGR)
BALL_PRED_COLOR = (0, 140, 255)  # orange
HUD_BG = (20, 20, 20)
WHITE = (255, 255, 255)


def id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread colour per track id (golden-ratio hue walk)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class Visualizer:
    def __init__(self, cfg: VisualizationConfig):
        self.cfg = cfg
        self._trails: dict[int, deque[tuple[int, tuple[float, float]]]] = {}

    def _update_trails(self, frame_idx: int, players: Iterable[TrackedObject]) -> None:
        maxlen = self.cfg.player_trajectory_length
        for p in players:
            self._trails.setdefault(p.track_id, deque(maxlen=maxlen)).append((frame_idx, p.bottom_center))
        # Forget tracks that have not been seen for a full trail length.
        stale = [tid for tid, d in self._trails.items() if frame_idx - d[-1][0] > maxlen]
        for tid in stale:
            del self._trails[tid]

    def draw(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        players: list[TrackedObject],
        ball: Optional[TrackedObject],
        ball_trajectory: Iterable[TrajectoryPoint] = (),
    ) -> np.ndarray:
        cfg = self.cfg
        out = frame.copy()
        t = cfg.line_thickness
        self._update_trails(frame_idx, players)

        if cfg.draw_player_trajectories:
            for tid, trail in self._trails.items():
                pts = [pt for _, pt in trail]
                if len(pts) >= 2:
                    arr = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(out, [arr], False, id_color(tid), max(1, t - 1), cv2.LINE_AA)

        for p in players:
            color = id_color(p.track_id)
            x1, y1, x2, y2 = (int(round(v)) for v in p.xyxy)
            if cfg.draw_player_boxes:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
            if cfg.draw_player_ids:
                self._label(out, f"#{p.track_id}", (x1, y1), color)

        if cfg.draw_ball and cfg.draw_ball_trajectory:
            self._draw_ball_trail(out, list(ball_trajectory))
        if cfg.draw_ball and ball is not None and (ball.status != PREDICTED or cfg.show_predicted_ball):
            cx, cy = (int(round(v)) for v in ball.center)
            radius = max(6, int(max(ball.xyxy[2] - ball.xyxy[0], ball.xyxy[3] - ball.xyxy[1]) / 2) + 4)
            if ball.status == PREDICTED:
                cv2.circle(out, (cx, cy), radius, BALL_PRED_COLOR, t, cv2.LINE_AA)
            else:
                cv2.circle(out, (cx, cy), radius, BALL_COLOR, t + 1, cv2.LINE_AA)
            # A small marker above the ball stays visible even when the ball is tiny.
            tri = np.array([[cx, cy - radius - 4], [cx - 7, cy - radius - 16], [cx + 7, cy - radius - 16]], np.int32)
            cv2.fillPoly(out, [tri], BALL_PRED_COLOR if ball.status == PREDICTED else BALL_COLOR, cv2.LINE_AA)

        if cfg.draw_hud:
            ball_state = "lost" if ball is None else ball.status
            self._hud(out, f"frame {frame_idx}  t={timestamp_s:7.2f}s  players {len(players)}  ball {ball_state}")
        return out

    def _draw_ball_trail(self, img: np.ndarray, traj: list[TrajectoryPoint]) -> None:
        if len(traj) < 2:
            return
        n = len(traj)
        for k in range(1, n):
            a, b = traj[k - 1], traj[k]
            if b.frame_idx - a.frame_idx > 1:
                continue  # don't connect across gaps in the history
            color = BALL_PRED_COLOR if b.status == PREDICTED else BALL_COLOR
            thick = max(1, int(round(1 + 2 * k / n)))  # older segments thinner
            cv2.line(img, (int(a.x), int(a.y)), (int(b.x), int(b.y)), color, thick, cv2.LINE_AA)

    def _label(self, img: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        fs = self.cfg.font_scale
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        x, y = origin
        y_top = max(0, y - th - base - 2)
        cv2.rectangle(img, (x, y_top), (x + tw + 4, y_top + th + base + 2), color, -1)
        cv2.putText(img, text, (x + 2, y_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)

    def _hud(self, img: np.ndarray, text: str) -> None:
        fs = self.cfg.font_scale * 1.1
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        h = min(img.shape[0], th + base + 14)
        w = min(img.shape[1], tw + 16)
        roi = img[:h, :w]
        bg = np.full_like(roi, HUD_BG)
        cv2.addWeighted(bg, 0.6, roi, 0.4, 0, dst=roi)
        cv2.putText(img, text, (8, th + 7), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, 1, cv2.LINE_AA)
