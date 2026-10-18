"""Synthetic soccer-like video with known ground truth.

Used by the smoke test and by ``scripts/make_synthetic_video.py``. It is not
meant to look realistic; it gives a deterministic clip with moving "players"
(coloured rectangles) and a small "ball" so the pipeline's video I/O,
tracking, drawing, and CSV export can be checked without real footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SyntheticScene:
    width: int
    height: int
    fps: float
    num_frames: int
    player_boxes: list[np.ndarray]  # per frame: (P, 4) xyxy
    ball_boxes: list[np.ndarray]  # per frame: (4,) xyxy


def generate_scene(
    width: int = 640, height: int = 360, fps: float = 25.0, num_frames: int = 60, num_players: int = 4, seed: int = 0
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    pw, ph = 18, 44
    # Players start spread across the pitch and walk with constant velocity,
    # bouncing off the borders.
    pos = np.stack(
        [np.linspace(60, width - 60 - pw, num_players), rng.uniform(60, height - 60 - ph, num_players)], axis=1
    )
    vel = rng.uniform(-3, 3, size=(num_players, 2))
    ball = np.array([width / 2, height / 2], dtype=float)
    bvel = np.array([5.0, 2.5])
    r = 5

    players, balls = [], []
    for _ in range(num_frames):
        players.append(np.concatenate([pos, pos + [pw, ph]], axis=1).astype(np.float32))
        balls.append(np.array([ball[0] - r, ball[1] - r, ball[0] + r, ball[1] + r], dtype=np.float32))
        pos += vel
        for k in range(2):
            lim = (width - pw) if k == 0 else (height - ph)
            hit = (pos[:, k] < 10) | (pos[:, k] > lim - 10)
            vel[hit, k] *= -1
        ball += bvel
        for k, lim in ((0, width), (1, height)):
            if ball[k] < 20 or ball[k] > lim - 20:
                bvel[k] *= -1
    return SyntheticScene(width, height, fps, num_frames, players, balls)


def render_frame(scene: SyntheticScene, i: int) -> np.ndarray:
    img = np.zeros((scene.height, scene.width, 3), np.uint8)
    img[:] = (40, 140, 40)  # grass (BGR)
    cv2.line(img, (scene.width // 2, 0), (scene.width // 2, scene.height), (230, 230, 230), 2)
    cv2.circle(img, (scene.width // 2, scene.height // 2), 50, (230, 230, 230), 2)
    colors = [(0, 0, 220), (220, 60, 0), (0, 0, 220), (220, 60, 0)]
    for k, (x1, y1, x2, y2) in enumerate(scene.player_boxes[i]):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[k % len(colors)], -1)
    x1, y1, x2, y2 = scene.ball_boxes[i]
    cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), int((x2 - x1) / 2), (255, 255, 255), -1)
    return img


def write_video(scene: SyntheticScene, path: str | Path, fourcc: str = "mp4v") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), scene.fps, (scene.width, scene.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {path} with codec {fourcc}")
    try:
        for i in range(scene.num_frames):
            writer.write(render_frame(scene, i))
    finally:
        writer.release()
    return path
