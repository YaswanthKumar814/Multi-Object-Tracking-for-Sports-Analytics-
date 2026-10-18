import csv

import numpy as np

from soccer_tracker.analytics import CSV_COLUMNS, RunStats, TrackCsvWriter
from soccer_tracker.tracking import BALL, DETECTED, PLAYER, PREDICTED, TrackedObject


def test_csv_rows_and_columns(tmp_path):
    p = tmp_path / "sub" / "t.csv"
    objs = [
        TrackedObject(PLAYER, 3, np.array([10, 20, 30, 60.0]), "person", 0.87, DETECTED),
        TrackedObject(BALL, 0, np.array([100, 100, 108, 108.0]), "sports ball", None, PREDICTED),
    ]
    with TrackCsvWriter(p) as w:
        w.write_frame(7, 0.28, objs)
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["object_type"] == "player" and rows[0]["track_id"] == "3"
    assert rows[0]["center_x"] == "20.0" and rows[0]["center_y"] == "40.0"
    assert rows[1]["confidence"] == "" and rows[1]["status"] == "predicted"
    assert rows[1]["timestamp_s"] == "0.2800" and rows[1]["frame"] == "7"


def test_run_stats_summary():
    s = RunStats()
    p = TrackedObject(PLAYER, 1, np.zeros(4), "person", 0.9)
    b = TrackedObject(BALL, 0, np.zeros(4), "sports ball", 0.5)
    s.record([p], b)
    s.record([p], None)
    s.elapsed_s = 1.0
    d = s.to_dict()
    assert d["frames_processed"] == 2
    assert d["ball"]["frames_detected"] == 1 and d["ball"]["frames_lost"] == 1
    assert d["ball"]["coverage"] == 0.5
    assert d["players"]["unique_track_ids"] == 1
