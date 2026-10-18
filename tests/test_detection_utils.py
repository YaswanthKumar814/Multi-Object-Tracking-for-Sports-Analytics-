import numpy as np
import pytest

from soccer_tracker.config import BallFilterConfig, PitchMaskConfig, PlayerFilterConfig
from soccer_tracker.detection import Detections, PitchMasker, box_iou, compute_tiles, filter_balls, filter_players, nms


def dets(boxes, conf=None, name="person"):
    boxes = np.asarray(boxes, float)
    conf = np.full(len(boxes), 0.9) if conf is None else np.asarray(conf, float)
    return Detections(boxes, conf, np.array([name] * len(boxes), dtype=object))


def test_detections_length_mismatch_raises():
    with pytest.raises(ValueError):
        Detections(np.zeros((2, 4)), np.zeros(3), np.array(["a", "b"], dtype=object))


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert box_iou(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0], abs=1e-6)
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_nms_keeps_best_and_is_class_aware():
    d = Detections(
        np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], float),
        np.array([0.5, 0.9, 0.7]),
        np.array(["person", "person", "sports ball"], dtype=object),
    )
    out = nms(d, 0.5)
    assert len(out) == 2
    assert set(out.class_name) == {"person", "sports ball"}
    assert 0.9 in out.confidence


@pytest.mark.parametrize("w,h", [(1920, 1080), (640, 360), (500, 300)])
def test_tiles_cover_frame(w, h):
    tiles = compute_tiles(w, h, 640, 0.2)
    cover = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        cover[y1:y2, x1:x2] = True
    assert cover.all()


def test_player_filter_size_and_aspect():
    d = dets([[0, 0, 20, 50], [0, 0, 5, 5], [0, 0, 100, 20], [0, 0, 50, 400]])
    out = filter_players(d, (500, 800, 3), PlayerFilterConfig(min_height_px=12, max_height_frac=0.6))
    # keeps the normal player; drops tiny, wide, and too-tall boxes
    assert len(out) == 1 and out.xyxy[0, 3] == 50


def test_ball_filter():
    d = dets([[0, 0, 8, 8], [0, 0, 200, 200], [0, 0, 30, 5]], name="sports ball")
    out = filter_balls(d, BallFilterConfig(max_size_px=80, max_aspect_ratio=2))
    assert len(out) == 1


def test_pitch_mask_filter_drops_players_off_grass():
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[100:] = (40, 140, 40)  # bottom half grass, top half "stands"
    mask = PitchMasker(PitchMaskConfig(dilate_px=0)).compute(frame)
    d = dets([[50, 120, 70, 170], [50, 10, 70, 60]])
    out = filter_players(d, frame.shape, PlayerFilterConfig(use_pitch_mask=True), mask)
    assert len(out) == 1 and out.xyxy[0, 1] == 120


def test_shared_pass_runs_once_per_frame_and_splits_classes():
    # yolo.py imports ultralytics lazily, so this runs without it installed.
    from soccer_tracker.config import DetectorConfig
    from soccer_tracker.detection.yolo import ClassView, _SharedPass, can_share_inference

    class Counting:
        calls = 0

        def detect(self, frame):
            Counting.calls += 1
            return Detections(
                np.array([[0, 0, 20, 50], [5, 5, 12, 12], [30, 30, 38, 38]], float),
                np.array([0.9, 0.05, 0.4]),
                np.array(["person", "sports ball", "sports ball"], dtype=object),
            )

    shared = _SharedPass(Counting())
    players = ClassView(shared, ["person"], 0.15)
    ball = ClassView(shared, ["Sports Ball"], 0.1)
    frame = np.zeros((10, 10, 3), np.uint8)
    p, b = players.detect(frame), ball.detect(frame)
    assert Counting.calls == 1
    assert list(p.class_name) == ["person"]
    assert len(b) == 1 and b.confidence[0] == pytest.approx(0.4)  # 0.05 is below the ball threshold
    players.detect(frame.copy())  # a new frame triggers a new pass
    assert Counting.calls == 2

    a, c = DetectorConfig(), DetectorConfig(classes=["sports ball"])
    assert can_share_inference(a, c)
    c.tiling.enabled = True
    assert not can_share_inference(a, c)
