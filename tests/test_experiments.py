"""Tracker factory, detection cache and the experiment plumbing (no model weights needed)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from soccer_tracker.cli import build_parser
from soccer_tracker.config import AppConfig, ByteTrackConfig, config_from_dict
from soccer_tracker.detection.base import Detections
from soccer_tracker.errors import ConfigError
from soccer_tracker.experiments.common import read_csv, system_info, write_csv
from soccer_tracker.experiments.detections import DetectionCache, compute_detections
from soccer_tracker.experiments.players import DEFAULT_PARAMS, grid_configs, run_tracker
from soccer_tracker.tracking import ALGORITHMS, PlayerTracker, build_mot_tracker
from soccer_tracker.tracking.factory import SHARED_PARAMS, TRACKER_LABELS, accepted_params


# --------------------------------------------------------------------------- #
# tracker factory / selectable algorithm
# --------------------------------------------------------------------------- #
def test_factory_builds_every_algorithm_with_shared_params():
    assert set(ALGORITHMS) == {"sort", "ocsort", "bytetrack"} == set(TRACKER_LABELS)
    for algo in ALGORITHMS:
        tracker = build_mot_tracker(algo, 25.0, **DEFAULT_PARAMS)
        assert hasattr(tracker, "update")
        assert set(accepted_params(algo)) <= set(SHARED_PARAMS)
    assert "proposed" in TRACKER_LABELS["bytetrack"]


def test_factory_rejects_unknown_names():
    with pytest.raises(ConfigError, match="Unknown tracker"):
        build_mot_tracker("deepsort", 25.0)
    with pytest.raises(ConfigError, match="Unknown tracker parameter"):
        build_mot_tracker("sort", 25.0, not_a_parameter=1)


@pytest.mark.parametrize("algorithm", ["sort", "ocsort", "bytetrack"])
def test_player_tracker_keeps_ids_with_each_algorithm(scene, algorithm):
    cfg = ByteTrackConfig(algorithm=algorithm, minimum_consecutive_frames=1)
    tracker = PlayerTracker(cfg, frame_rate=scene.fps)
    ids_per_frame = []
    for boxes in scene.player_boxes:
        out = tracker.update(Detections(boxes, np.full(len(boxes), 0.9), np.array(["person"] * len(boxes), object)))
        ids_per_frame.append(sorted(o.track_id for o in out))
    settled = ids_per_frame[5:]
    assert all(len(ids) == len(scene.player_boxes[0]) for ids in settled)
    assert all(ids == settled[0] for ids in settled)  # no identity switches on easy data


def test_algorithm_is_validated_and_exposed_on_the_cli():
    assert AppConfig().players.tracker.algorithm == "bytetrack"
    assert config_from_dict({"players": {"tracker": {"algorithm": "ocsort"}}}).players.tracker.algorithm == "ocsort"
    with pytest.raises(ConfigError, match="algorithm"):
        config_from_dict({"players": {"tracker": {"algorithm": "kalman"}}})
    assert build_parser().parse_args(["-i", "x.mp4", "--tracker", "sort"]).tracker == "sort"


# --------------------------------------------------------------------------- #
# detection cache
# --------------------------------------------------------------------------- #
class _FakeSeq:
    """Minimal stand-in for a dataset sequence."""

    def __init__(self, scene):
        from soccer_tracker.datasets.base import SequenceInfo

        self.scene = scene
        self.info = SequenceInfo("Synthetic", "scene", "none", "test", scene.fps, scene.width, scene.height,
                                 scene.num_frames, 1, "memory")

    def frames(self):
        from soccer_tracker.synthetic import render_frame

        for i in range(self.scene.num_frames):
            yield i + 1, render_frame(self.scene, i)


@pytest.fixture
def cache(scene, gt_factory):
    cfg = AppConfig()
    return compute_detections(_FakeSeq(scene), cfg, detectors=gt_factory(cfg, ball_drop=(10, 11, 12)))


def test_cache_holds_per_frame_detections(cache, scene):
    assert cache.num_frames == scene.num_frames
    assert len(cache.players(1)) == 4 and len(cache.balls(1)) == 1
    np.testing.assert_allclose(cache.players(7).xyxy, scene.player_boxes[6], atol=1e-4)
    assert len(cache.balls(11)) == 0  # frame index 10 was dropped by the fake detector
    assert len(cache.players(scene.num_frames + 5)) == 0
    assert list(cache.players(3).class_name) == ["person"] * 4
    t = cache.detector_time()
    assert t["mean_ms"] >= 0 and t["total_s"] >= 0


def test_cache_round_trips_through_disk(cache, tmp_path):
    path = tmp_path / "sub" / "c.npz"
    cache.save(path)
    loaded = DetectionCache.load(path)
    assert loaded.meta == cache.meta and loaded.num_frames == cache.num_frames
    np.testing.assert_array_equal(loaded.p_xyxy, cache.p_xyxy)
    np.testing.assert_array_equal(loaded.balls(5).xyxy, cache.balls(5).xyxy)


# --------------------------------------------------------------------------- #
# tracker comparison plumbing
# --------------------------------------------------------------------------- #
def test_grid_gives_every_tracker_the_same_budget():
    configs = grid_configs()
    assert len(configs) == 48
    assert all(set(c) == set(DEFAULT_PARAMS) for c in configs)
    assert all(c["track_activation_threshold"] == c["high_conf_det_threshold"] for c in configs)
    assert len({tuple(sorted(c.items())) for c in configs}) == 48


@pytest.mark.parametrize("algorithm", ["sort", "ocsort", "bytetrack"])
def test_run_tracker_replays_cached_detections(cache, scene, algorithm):
    cfg = AppConfig()
    params = dict(DEFAULT_PARAMS, minimum_consecutive_frames=1)
    rows, seconds = run_tracker(algorithm, params, cache, _FakeSeq(scene), cfg.players.filter, 0.15)
    assert rows.shape[1] == 7 and seconds > 0
    assert len(np.unique(rows[:, 1])) == 4  # four players, four identities
    last = rows[rows[:, 0] == scene.num_frames]
    assert len(last) == 4
    # boxes are written as x, y, w, h
    np.testing.assert_allclose(np.sort(last[:, 4]), np.full(4, 18.0), atol=1e-3)
    # replaying the same cache is deterministic
    again, _ = run_tracker(algorithm, params, cache, _FakeSeq(scene), cfg.players.filter, 0.15)
    np.testing.assert_array_equal(rows, again)


def test_ball_methods_separate_detected_from_predicted(cache, scene):
    from soccer_tracker.experiments.ball import run_ball_methods

    outputs, candidates, ms = run_ball_methods(cache, _FakeSeq(scene), AppConfig())
    assert set(outputs) == {"detection_only", "kalman_detected", "kalman", "kalman_motion"}
    # the detector dropped frames 11-13: detection-only has nothing there, the Kalman tracker predicts
    for f in (11, 12, 13):
        assert f not in outputs["detection_only"] and f not in outputs["kalman_detected"]
        assert outputs["kalman"][f].status == "predicted" and outputs["kalman"][f].confidence is None
    assert outputs["kalman"][20].status == "detected"
    assert all(o.status == "detected" for o in outputs["kalman_detected"].values())
    assert len(candidates) == scene.num_frames and all(v >= 0 for v in ms.values())
    # the proposed tracker (Kalman + motion flights) covers at least what the Kalman tracker does here
    assert outputs["kalman_motion"][20].status == "detected"
    assert set(outputs["kalman"]) <= set(outputs["kalman_motion"])


# --------------------------------------------------------------------------- #
# result files and database
# --------------------------------------------------------------------------- #
def test_csv_round_trip_restores_types(tmp_path):
    rows = [{"name": "a", "n": 3, "x": 1.5, "flag": True, "none": None}, {"name": "b", "n": 4, "x": 2.0}]
    write_csv(tmp_path / "r.csv", rows)
    back = read_csv(tmp_path / "r.csv")
    assert back[0] == {"name": "a", "n": 3, "x": 1.5, "flag": True, "none": None}
    assert back[1]["flag"] is None and back[1]["x"] == 2.0


def test_system_info_records_hardware():
    info = system_info("cpu")
    assert info["device"] == "cpu" and info["cpu_logical_cores"] >= 1 and info["cpu"]


def test_database_schema_is_valid_sql():
    from soccer_tracker.experiments.database import SCHEMA

    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"experiment_runs", "datasets", "sequences", "tracking_metrics", "detection_metrics",
            "track_summaries"} <= tables
    conn.close()
