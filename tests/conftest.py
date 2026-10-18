from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soccer_tracker.config import AppConfig
from soccer_tracker.detection.base import Detections
from soccer_tracker.synthetic import generate_scene, write_video


class GroundTruthDetector:
    """Stand-in detector that replays known boxes frame by frame.

    Lets the pipeline be tested end to end without model weights. Frames in
    ``drop_frames`` return no detections, to exercise tracker gap handling.
    """

    def __init__(self, boxes_per_frame, class_name: str, confidence: float = 0.9, drop_frames=()):
        self.boxes = boxes_per_frame
        self.class_name = class_name
        self.confidence = confidence
        self.drop = set(drop_frames)
        self.i = 0

    def detect(self, frame: np.ndarray) -> Detections:
        i = self.i
        self.i += 1
        if i in self.drop or i >= len(self.boxes):
            return Detections.empty()
        b = np.asarray(self.boxes[i], dtype=np.float32).reshape(-1, 4)
        return Detections(b, np.full(len(b), self.confidence), np.array([self.class_name] * len(b), dtype=object))


@pytest.fixture
def scene():
    return generate_scene(num_frames=40, num_players=4, seed=0)


@pytest.fixture
def synthetic_video(tmp_path: Path, scene) -> Path:
    return write_video(scene, tmp_path / "synthetic.mp4")


@pytest.fixture
def gt_factory(scene):
    """Detector factory for Pipeline(...) that uses ground-truth boxes."""

    def factory(cfg: AppConfig, ball_drop=(10, 11, 12)):
        players = GroundTruthDetector(scene.player_boxes, "person")
        ball = GroundTruthDetector(scene.ball_boxes, "sports ball", 0.8, drop_frames=ball_drop) if cfg.ball.enabled else None
        return players, ball

    return factory


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.output_video = str(tmp_path / "out" / "annotated.mp4")
    cfg.io.output_csv = str(tmp_path / "out" / "tracks.csv")
    cfg.io.output_summary = str(tmp_path / "out" / "summary.json")
    cfg.io.log_every = 10
    cfg.players.tracker.minimum_consecutive_frames = 1
    return cfg
