"""Checks against real Ultralytics weights.

Skipped unless ultralytics is installed AND the weights file exists (set
SOCCER_TRACKER_WEIGHTS to point elsewhere). Nothing is downloaded here.
Run explicitly with:  pytest -m integration
"""

import os
from pathlib import Path

import numpy as np
import pytest

from soccer_tracker.config import DetectorConfig
from soccer_tracker.errors import ModelLoadError

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("SOCCER_TRACKER_WEIGHTS", ROOT / "weights" / "yolo11n.pt"))

ultralytics = pytest.importorskip("ultralytics")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not WEIGHTS.is_file(), reason=f"weights not found at {WEIGHTS}"),
]


def test_loads_and_resolves_classes():
    from soccer_tracker.detection.yolo import ModelCache, YoloDetector

    cache = ModelCache()
    p = YoloDetector(DetectorConfig(weights=str(WEIGHTS), classes=["person"], device="cpu"), cache)
    b = YoloDetector(DetectorConfig(weights=str(WEIGHTS), classes=["sports ball"], device="cpu"), cache)
    assert p.model is b.model  # loaded once
    assert p.class_ids == [0] and b.class_ids == [32]


def test_unknown_class_name_is_reported():
    from soccer_tracker.detection.yolo import YoloDetector

    with pytest.raises(ModelLoadError, match="not found in model"):
        YoloDetector(DetectorConfig(weights=str(WEIGHTS), classes=["referee"], device="cpu"))


def test_missing_custom_weights_is_reported(tmp_path):
    from soccer_tracker.detection.yolo import load_yolo

    with pytest.raises(ModelLoadError, match="not an official Ultralytics asset"):
        load_yolo(str(tmp_path / "my_ball_model.pt"))


def test_official_weights_without_download_permission(tmp_path):
    from soccer_tracker.detection.yolo import load_yolo

    with pytest.raises(ModelLoadError, match="allow_download is false"):
        load_yolo(str(tmp_path / "yolo11n.pt"), allow_download=False)


@pytest.mark.parametrize("tiling", [False, True])
def test_detect_returns_valid_detections(tiling):
    from soccer_tracker.detection.yolo import YoloDetector

    cfg = DetectorConfig(weights=str(WEIGHTS), classes=["person", "sports ball"], device="cpu", imgsz=640)
    cfg.tiling.enabled = tiling
    det = YoloDetector(cfg)
    frame = np.full((720, 1280, 3), (40, 140, 40), np.uint8)
    out = det.detect(frame)
    assert out.xyxy.shape[1] == 4 and len(out.confidence) == len(out)


def test_pipeline_with_real_model(synthetic_video, test_config):
    from soccer_tracker.pipeline import Pipeline

    for d in (test_config.players.detector, test_config.ball.detector):
        d.weights = str(WEIGHTS)
        d.device = "cpu"
        d.imgsz = 640
    test_config.io.max_frames = 5
    summary = Pipeline(test_config).run(synthetic_video)
    assert summary["frames_processed"] == 5
    assert Path(test_config.io.output_csv).is_file()


def test_shared_inference_factory_used_by_default(test_config):
    from soccer_tracker.detection.yolo import ClassView
    from soccer_tracker.pipeline import build_yolo_detectors

    for d in (test_config.players.detector, test_config.ball.detector):
        d.weights = str(WEIGHTS)
    p, b = build_yolo_detectors(test_config)
    assert isinstance(p, ClassView) and isinstance(b, ClassView)
    test_config.ball.share_player_inference = False
    p, b = build_yolo_detectors(test_config)
    assert not isinstance(p, ClassView)
