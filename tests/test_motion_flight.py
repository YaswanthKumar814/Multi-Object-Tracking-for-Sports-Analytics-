"""Motion-based ball candidates, flight linking, and flight adoption by the ball tracker."""

import numpy as np
import pytest

from soccer_tracker.config import BallTrackerConfig, MotionBallConfig
from soccer_tracker.detection import Detections
from soccer_tracker.detection.motion import MotionBallDetector
from soccer_tracker.tracking import DETECTED, BallTracker
from soccer_tracker.tracking.ball_tracker import MOTION_CLASS
from soccer_tracker.tracking.flight import FlightLinker

import cv2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def textured_background(h=360, w=640, seed=0):
    """Dark-green 'pitch' with enough texture for ORB to estimate camera motion."""
    rng = np.random.default_rng(seed)
    base = np.zeros((h + 200, w + 200, 3), np.uint8)
    base[:] = (40, 110, 40)
    for _ in range(400):
        x, y = int(rng.integers(0, w + 200)), int(rng.integers(0, h + 200))
        c = tuple(int(v) for v in rng.integers(0, 120, 3))
        cv2.rectangle(base, (x, y), (x + int(rng.integers(3, 15)), y + int(rng.integers(3, 15))), c, -1)
    return base


def frame_at(base, t, ball_xy, pan_px=0, h=360, w=640):
    ox = 100 + pan_px * t  # camera pans: the background shifts every frame
    img = base[100:100 + h, ox:ox + w].copy()
    if ball_xy is not None:
        cv2.circle(img, (int(ball_xy[0]), int(ball_xy[1])), 5, (255, 255, 255), -1)
    return img


def pts(*xy, conf=0.2):
    if not xy:
        return Detections.empty()
    boxes = [[x - 3, y - 3, x + 3, y + 3] for x, y in xy]
    return Detections(np.array(boxes, float), np.full(len(xy), conf), np.array([MOTION_CLASS] * len(xy), dtype=object))


def ball(x, y, conf=0.8):
    return Detections(np.array([[x - 4, y - 4, x + 4, y + 4]], float), np.array([conf]), np.array(["sports ball"], dtype=object))


# --------------------------------------------------------------------------- #
# MotionBallDetector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pan", [0, 3])
def test_motion_detector_finds_moving_ball_with_camera_pan(pan):
    base = textured_background()
    det = MotionBallDetector(MotionBallConfig())
    hits = 0
    for t in range(12):
        xy = (100 + 25 * t, 300 - 15 * t)
        out = det.detect(frame_at(base, t, xy, pan_px=pan))
        if t < 2:
            assert len(out) == 0  # needs two past frames
            continue
        d = np.linalg.norm(out.centers - np.array(xy), axis=1) if len(out) else np.array([np.inf])
        hits += bool(d.min() < 6)
        assert len(out) <= 3, "camera motion should be compensated, not seen as motion"
    assert hits >= 9


def test_static_ball_under_camera_pan_never_becomes_a_flight():
    # Still on the pitch while the camera pans 2 px/frame: it drifts in the image.
    # Sub-pixel alignment error can leave slivers at its edge; they must not link
    # into a flight.
    base = textured_background()
    det = MotionBallDetector(MotionBallConfig())
    linker = FlightLinker(MotionBallConfig())
    flights = []
    for t in range(30):
        flights += linker.update(t, det.detect(frame_at(base, t, (200 - 2 * t, 200), pan_px=2)))
    assert not flights


# --------------------------------------------------------------------------- #
# FlightLinker
# --------------------------------------------------------------------------- #
def run_linker(paths, frames, noise=0, seed=1):
    rng = np.random.default_rng(seed)
    linker = FlightLinker(MotionBallConfig())
    confirmed = {}
    for t in range(frames):
        xy = [p(t) for p in paths if p(t) is not None]
        xy += [tuple(rng.uniform(0, 600, 2)) for _ in range(noise)]
        for fl in linker.update(t, pts(*xy)):
            confirmed[fl.id] = fl
    return confirmed


def test_fast_straight_path_becomes_a_flight():
    flights = run_linker([lambda t: (50 + 20 * t, 400 - 12 * t)], 15, noise=15)
    assert flights
    fl = max(flights.values(), key=len)
    assert len(fl) >= 12 and fl.first_frame <= 1


def test_slow_or_swinging_paths_are_not_flights():
    slow = lambda t: (100 + 3 * t, 200)  # a jogging player
    swing = lambda t: (300 + 20 * (1 if t % 2 else -1), 300 + 5 * t)  # a limb going back and forth
    assert not run_linker([slow, swing], 20)


# --------------------------------------------------------------------------- #
# BallTracker + flights
# --------------------------------------------------------------------------- #
def kick_scenario(launch_offset=(0.0, 0.0), yolo_during_flight=None):
    """Ball seen at rest at (100, 300) for frames 0-9, then kicked at 25 px/frame.

    The flight (motion blobs) starts at frame 11. ``launch_offset`` shifts the
    whole flight, so it no longer comes from the ball.
    """
    cfg = BallTrackerConfig(min_hits=2)
    tracker = BallTracker(cfg)
    linker = FlightLinker(MotionBallConfig())
    outs, amends = {}, {}
    for t in range(30):
        yolo = ball(100, 300) if t < 10 else Detections.empty()
        if yolo_during_flight is not None and t >= 10:
            yolo = yolo_during_flight(t)
        k = t - 10  # frames since the kick
        moving = pts((100 + 25 * k + launch_offset[0], 300 - 15 * k + launch_offset[1])) if k >= 1 else pts()
        outs[t] = tracker.update(t, yolo, (720, 1280), linker.update(t, moving))
        amends.update(tracker.take_amendments())
    return outs, amends


def test_flight_launched_from_the_ball_is_followed_and_backfilled():
    outs, amends = kick_scenario()
    late = outs[25]
    assert late is not None and late.status == DETECTED and late.class_name == MOTION_CLASS
    assert late.center == pytest.approx((100 + 25 * 15, 300 - 15 * 15), abs=3)
    # frames before the flight was confirmed are filled in retroactively
    assert 11 in amends and amends[11].center == pytest.approx((125, 285), abs=3)
    assert outs[25].track_id == outs[5].track_id  # same ball, same track


def test_flight_not_launched_from_the_ball_is_ignored():
    outs, amends = kick_scenario(launch_offset=(0, -150))  # same motion, 150 px away: a player, not the ball
    assert all(o is None or o.class_name != MOTION_CLASS for o in outs.values())
    assert not amends


def test_confident_detection_beats_a_flight():
    # The detector keeps seeing the ball at rest; a fast blob nearby must not steal it.
    outs, _ = kick_scenario(yolo_during_flight=lambda t: ball(100, 300))
    assert all(o is not None and o.class_name == "sports ball" for o in outs.values() if o is not None)
    assert outs[25].center == pytest.approx((100, 300), abs=2)
