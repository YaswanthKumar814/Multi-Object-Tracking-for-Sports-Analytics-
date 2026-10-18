import numpy as np
import pytest

from soccer_tracker.config import BallTrackerConfig
from soccer_tracker.detection import Detections
from soccer_tracker.tracking import DETECTED, PREDICTED, BallTracker


def ball_at(*pts, conf=0.6, r=4):
    if not pts:
        return Detections.empty()
    boxes = [[x - r, y - r, x + r, y + r] for x, y in pts]
    confs = conf if isinstance(conf, (list, tuple)) else [conf] * len(pts)
    return Detections(np.array(boxes, float), np.array(confs, float), np.array(["sports ball"] * len(pts), dtype=object))


def linear(i, x0=100.0, y0=100.0, vx=6.0, vy=3.0):
    return x0 + vx * i, y0 + vy * i


def test_not_reported_until_min_hits():
    t = BallTracker(BallTrackerConfig(min_hits=3))
    assert t.update(0, ball_at(linear(0))) is None
    assert t.update(1, ball_at(linear(1))) is None
    out = t.update(2, ball_at(linear(2)))
    assert out is not None and out.status == DETECTED


def test_follows_linear_motion():
    t = BallTracker(BallTrackerConfig(min_hits=1))
    for i in range(20):
        out = t.update(i, ball_at(linear(i)))
    assert out.center == pytest.approx(linear(19), abs=1.5)
    assert t.tracks_started == 1


def test_coasts_through_gap_and_predicts_motion():
    t = BallTracker(BallTrackerConfig(min_hits=1, max_missed_frames=10))
    for i in range(15):
        t.update(i, ball_at(linear(i)))
    outs = [t.update(i, Detections.empty()) for i in range(15, 20)]
    assert all(o is not None and o.status == PREDICTED and o.confidence is None for o in outs)
    # prediction should continue along the trajectory
    assert outs[-1].center == pytest.approx(linear(19), abs=6)
    # re-associates when the ball reappears where expected, same track id
    back = t.update(20, ball_at(linear(20)))
    assert back.status == DETECTED and back.track_id == outs[0].track_id


def test_dropped_after_max_missed():
    t = BallTracker(BallTrackerConfig(min_hits=1, max_missed_frames=3))
    for i in range(5):
        t.update(i, ball_at(linear(i)))
    results = [t.update(i, Detections.empty()) for i in range(5, 10)]
    assert results[2] is not None  # 3rd miss still coasting
    assert results[3] is None and not t.active


def test_rejects_false_positive_outside_gate():
    t = BallTracker(BallTrackerConfig(min_hits=1, gate_px=40, reinit_confidence=0.95))
    for i in range(10):
        t.update(i, ball_at(linear(i)))
    # a weak far-away candidate (e.g. a white sock) must not hijack the track
    out = t.update(10, ball_at((500.0, 50.0), conf=0.3))
    assert out.status == PREDICTED
    assert out.center == pytest.approx(linear(10), abs=4)


def test_prefers_candidate_near_prediction():
    t = BallTracker(BallTrackerConfig(min_hits=1, gate_px=80))
    for i in range(10):
        t.update(i, ball_at(linear(i)))
    near = linear(10)
    out = t.update(10, ball_at((near[0] + 50, near[1]), near, conf=[0.7, 0.5]))
    assert out.center == pytest.approx(near, abs=3)


def test_reacquires_on_confident_far_detection():
    cfg = BallTrackerConfig(min_hits=1, gate_px=40, gate_growth_px=0, reinit_confidence=0.6, reinit_after_missed=2)
    t = BallTracker(cfg)
    for i in range(10):
        t.update(i, ball_at(linear(i)))
    first_id = t.track_id
    far = (600.0, 300.0)
    assert t.update(10, ball_at(far, conf=0.9)).status == PREDICTED  # 1 miss: not yet
    out = t.update(11, ball_at(far, conf=0.9))  # 2 misses: re-initialise
    assert out.status == DETECTED and out.center == pytest.approx(far, abs=0.5)
    assert t.track_id != first_id and t.tracks_started == 2


def test_track_dropped_when_prediction_leaves_frame():
    t = BallTracker(BallTrackerConfig(min_hits=1, gate_px=10, gate_growth_px=0, max_missed_frames=50))
    for i in range(10):
        t.update(i, ball_at(linear(i, x0=560, vx=12)), frame_shape=(360, 640, 3))
    results = [t.update(i, Detections.empty(), frame_shape=(360, 640, 3)) for i in range(10, 30)]
    assert results[-1] is None and not t.active


def test_trajectory_is_bounded():
    t = BallTracker(BallTrackerConfig(min_hits=1, trajectory_length=5))
    for i in range(20):
        t.update(i, ball_at(linear(i)))
    assert len(t.trajectory) == 5 and t.trajectory[-1].frame_idx == 19
