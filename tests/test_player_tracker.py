import numpy as np

from soccer_tracker.config import ByteTrackConfig
from soccer_tracker.detection import Detections
from soccer_tracker.tracking import PlayerTracker


def frame_dets(i, conf=0.9):
    boxes = np.array([[10 + 3 * i, 20, 30 + 3 * i, 70], [300 - 2 * i, 40, 320 - 2 * i, 90]], float)
    return Detections(boxes, np.full(2, conf), np.array(["person", "person"], dtype=object))


def test_ids_stable_for_two_moving_players():
    t = PlayerTracker(ByteTrackConfig(minimum_consecutive_frames=2), frame_rate=25)
    history = {}
    for i in range(30):
        for obj in t.update(frame_dets(i)):
            history.setdefault(obj.track_id, []).append(obj.center[0])
    assert len(history) == 2
    left, right = sorted(history.values(), key=lambda xs: xs[0])
    assert np.all(np.diff(left) > 0) and np.all(np.diff(right) < 0)


def test_track_survives_short_gap():
    t = PlayerTracker(ByteTrackConfig(minimum_consecutive_frames=1, lost_track_buffer=30), frame_rate=25)
    ids_before = {o.track_id for i in range(10) for o in t.update(frame_dets(i))}
    for _ in range(5):
        assert t.update(Detections.empty()) == []
    ids_after = {o.track_id for o in t.update(frame_dets(15))}
    assert ids_after <= ids_before and len(ids_after) == 2


def test_low_confidence_detections_do_not_start_tracks():
    t = PlayerTracker(ByteTrackConfig(track_activation_threshold=0.5, minimum_consecutive_frames=1), frame_rate=25)
    out = [t.update(frame_dets(i, conf=0.2)) for i in range(5)]
    assert all(o == [] for o in out)


def test_class_name_is_preserved():
    t = PlayerTracker(ByteTrackConfig(minimum_consecutive_frames=1), frame_rate=25)
    d = Detections(np.array([[0, 0, 20, 50], [100, 0, 120, 50]], float), np.array([0.9, 0.9]),
                   np.array(["player", "goalkeeper"], dtype=object))
    for _ in range(3):
        out = t.update(d)
    assert sorted(o.class_name for o in out) == ["goalkeeper", "player"]
