"""SQLite store for experiment results: results/sports_analytics.db

The database is rebuilt from the CSV/JSON result files, which stay the source
of truth (they are easier to diff and to load in a spreadsheet). SQLite adds
ad-hoc querying, e.g.::

    SELECT tracker, hota, mota, idf1 FROM tracking_metrics
    WHERE scope = 'All test sequences' AND param_set = 'dev-tuned';

Tables: experiment_runs, datasets, sequences, tracking_metrics,
detection_metrics, track_summaries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from .common import read_csv, read_json
from .manifest import Manifest

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE experiment_runs (
    run_id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,              -- player_tracking | ball | model_comparison | demo
    created_at TEXT,
    detector TEXT,
    device TEXT,
    cpu TEXT,
    ram_gb REAL,
    git_commit TEXT,
    evaluator TEXT,
    config_json TEXT                  -- parameters / system information as JSON
);
CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    source_url TEXT,
    licence TEXT,
    annotation_format TEXT,
    used_for TEXT,
    sequences_used INTEGER,
    frames_used INTEGER
);
CREATE TABLE sequences (
    sequence_id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id),
    name TEXT NOT NULL,
    split TEXT,
    role TEXT,                        -- dev (tuning) | test
    fps REAL, width INTEGER, height INTEGER,
    frames_evaluated INTEGER,
    source_first_frame INTEGER, source_last_frame INTEGER,
    player_gt_tracks INTEGER, player_gt_boxes INTEGER, ball_visible_frames INTEGER,
    UNIQUE (dataset_id, name)
);
CREATE TABLE tracking_metrics (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES experiment_runs(run_id),
    scope TEXT NOT NULL,              -- 'sequence' or an aggregate group name
    sequence_id INTEGER REFERENCES sequences(sequence_id),
    tracker TEXT NOT NULL, tracker_label TEXT, param_set TEXT NOT NULL,
    frames INTEGER,
    hota REAL, deta REAL, assa REAL, mota REAL, idf1 REAL, motp REAL,
    precision_pct REAL, recall_pct REAL,
    id_switches INTEGER, false_positives INTEGER, false_negatives INTEGER, fragmentations INTEGER,
    mostly_tracked INTEGER, mostly_lost INTEGER,
    tracker_ms_per_frame REAL, detector_ms_per_frame REAL, end_to_end_fps REAL
);
CREATE TABLE detection_metrics (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES experiment_runs(run_id),
    scope TEXT NOT NULL,
    sequence_id INTEGER REFERENCES sequences(sequence_id),
    target TEXT NOT NULL,             -- player | ball
    method TEXT NOT NULL,
    criterion TEXT,
    tp INTEGER, fp INTEGER, fn INTEGER,
    precision_pct REAL, recall_pct REAL, f1_pct REAL, ap50_pct REAL,
    coverage_pct REAL, visible_coverage_pct REAL,
    frames_detected INTEGER, frames_predicted INTEGER, frames_lost INTEGER,
    center_error_mean_px REAL, track_segments INTEGER, track_breaks INTEGER, id_switches INTEGER
);
CREATE TABLE track_summaries (
    id INTEGER PRIMARY KEY,
    sequence_id INTEGER NOT NULL REFERENCES sequences(sequence_id),
    tracker TEXT NOT NULL, param_set TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    first_frame INTEGER, last_frame INTEGER, frames_tracked INTEGER,
    mean_confidence REAL, path_length_px REAL
);
"""

DATASET_INFO = {
    "SportsMOT": ("https://github.com/MCG-NJU/SportsMOT", "CC BY-NC 4.0 (research use)",
                  "MOT Challenge text: frame,id,x,y,w,h,1,1,1 (players only, no ball)",
                  "player tracking (dev + test)"),
    "UVY-Track": ("https://zenodo.org/records/21303900", "CC BY (videos collected under CC BY)",
                  "MOT-style text: frame,id,x,y,w,h,conf,class,visibility + labels.txt",
                  "player tracking and ball detection/tracking (test)"),
    "InFactory": ("https://huggingface.co/datasets/infactory-ai/ball-tracking", "CC BY-NC 4.0",
                  "YOLO txt per frame: class cx cy w h (normalised); visible / not_visible folders",
                  "ball detection only (test)"),
}


def _run(conn, stage: str, payload: dict | None) -> int:
    payload = payload or {}
    sysinfo = payload.get("system", {})
    cur = conn.execute(
        "INSERT INTO experiment_runs (stage, created_at, detector, device, cpu, ram_gb, git_commit, evaluator, config_json)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (stage, sysinfo.get("timestamp"), payload.get("detector"), sysinfo.get("device"), sysinfo.get("cpu"),
         sysinfo.get("ram_gb"), sysinfo.get("git_commit"), payload.get("evaluator"),
         json.dumps({k: v for k, v in payload.items() if k not in ("summary", "per_sequence")}, default=str)),
    )
    return int(cur.lastrowid)


def build_database(manifest: Manifest) -> Path:
    mdir = manifest.results_dir / "metrics"
    db_path = manifest.results_dir / "sports_analytics.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)

        # datasets and sequences
        seq_ids: dict[tuple[str, str], int] = {}
        seqs = manifest.all_sequences()
        for name, (url, lic, fmt, used) in DATASET_INFO.items():
            mine = [s for s in seqs if s.info.dataset == name]
            if not mine:
                continue
            cur = conn.execute(
                "INSERT INTO datasets (name, source_url, licence, annotation_format, used_for, sequences_used, frames_used)"
                " VALUES (?,?,?,?,?,?,?)",
                (name, url, lic, fmt, used, len(mine), sum(s.info.num_frames for s in mine)))
            did = cur.lastrowid
            for s in mine:
                d = s.describe()
                cur = conn.execute(
                    "INSERT INTO sequences (dataset_id, name, split, role, fps, width, height, frames_evaluated,"
                    " source_first_frame, source_last_frame, player_gt_tracks, player_gt_boxes, ball_visible_frames)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (did, d["sequence"], d["split"], d["role"], d["fps"], d["width"], d["height"],
                     d["frames_evaluated"], d["source_first_frame"], d["source_last_frame"], d["player_gt_tracks"],
                     d["player_gt_boxes"], d["ball_visible_frames"]))
                seq_ids[(name, d["sequence"])] = int(cur.lastrowid)

        # player tracking
        if (mdir / "player_tracking.json").is_file():
            payload = read_json(mdir / "player_tracking.json")
            run_id = _run(conn, "player_tracking", payload)

            def add_tracking(r: dict, scope: str, sid):
                conn.execute(
                    "INSERT INTO tracking_metrics (run_id, scope, sequence_id, tracker, tracker_label, param_set, frames,"
                    " hota, deta, assa, mota, idf1, motp, precision_pct, recall_pct, id_switches, false_positives,"
                    " false_negatives, fragmentations, mostly_tracked, mostly_lost, tracker_ms_per_frame,"
                    " detector_ms_per_frame, end_to_end_fps) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, scope, sid, r["tracker"], r["tracker_label"], r["param_set"], r["frames"], r["HOTA"],
                     r["DetA"], r["AssA"], r["MOTA"], r["IDF1"], r["MOTP"], r["CLR_Pr"], r["CLR_Re"], r["IDSW"],
                     r["CLR_FP"], r["CLR_FN"], r["Frag"], r["MT"], r["ML"], r["tracker_ms_per_frame"],
                     r["detector_ms_per_frame"], r["end_to_end_fps"]))

            for r in payload["per_sequence"]:
                add_tracking(r, "sequence", seq_ids.get((r["dataset"], r["sequence"])))
            for r in payload["summary"]:
                add_tracking(r, r["group"], None)

            # per-track summaries from the saved MOT files
            for s in seqs:
                sid = seq_ids.get((s.info.dataset, s.info.name))
                for set_name in payload["parameter_sets"]:
                    for tracker in payload["parameter_sets"][set_name]:
                        path = manifest.results_dir / "tracks" / set_name / tracker / f"{s.info.key}.txt"
                        if not path.is_file():
                            continue
                        rows = np.loadtxt(path, delimiter=",", ndmin=2)
                        for tid in np.unique(rows[:, 1]):
                            t = rows[rows[:, 1] == tid]
                            t = t[np.argsort(t[:, 0])]
                            foot = np.stack([t[:, 2] + t[:, 4] / 2, t[:, 3] + t[:, 5]], axis=1)
                            length = float(np.linalg.norm(np.diff(foot, axis=0), axis=1).sum()) if len(t) > 1 else 0.0
                            conn.execute(
                                "INSERT INTO track_summaries (sequence_id, tracker, param_set, track_id, first_frame,"
                                " last_frame, frames_tracked, mean_confidence, path_length_px) VALUES (?,?,?,?,?,?,?,?,?)",
                                (sid, tracker, set_name, int(tid), int(t[0, 0]), int(t[-1, 0]), len(t),
                                 round(float(t[:, 6].mean()), 4), round(length, 1)))

        def add_detection(run_id, scope, sid, target, method, criterion, r, p, rc, f1, ap=None):
            conn.execute(
                "INSERT INTO detection_metrics (run_id, scope, sequence_id, target, method, criterion, tp, fp, fn,"
                " precision_pct, recall_pct, f1_pct, ap50_pct, coverage_pct, visible_coverage_pct, frames_detected,"
                " frames_predicted, frames_lost, center_error_mean_px, track_segments, track_breaks, id_switches)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, scope, sid, target, method, criterion, r.get("tp"), r.get("fp"), r.get("fn"), p, rc, f1, ap,
                 r.get("coverage"), r.get("visible_coverage"), r.get("frames_detected"), r.get("frames_predicted"),
                 r.get("frames_lost"), r.get("center_error_mean_px"), r.get("track_segments"), r.get("track_breaks"),
                 r.get("id_switches")))

        # player detection (detector only)
        if (mdir / "player_detection.csv").is_file():
            run_id = _run(conn, "player_detection", read_json(mdir / "player_tracking.json")
                          if (mdir / "player_tracking.json").is_file() else None)
            for r in read_csv(mdir / "player_detection.csv"):
                add_detection(run_id, "sequence", seq_ids.get((r["dataset"], r["sequence"])), "player", r["method"],
                              "IoU >= 0.5, confidence >= 0.5", r, r["precision"], r["recall"], r["f1"], r["ap50"])

        # ball
        if (mdir / "ball.json").is_file():
            payload = read_json(mdir / "ball.json")
            run_id = _run(conn, "ball", payload)
            for rows, scoped in ((payload["per_sequence"], True), (payload["summary"], False)):
                for r in rows:
                    sid = seq_ids.get((r["dataset"], r["sequence"])) if scoped else None
                    scope = "sequence" if scoped else r["group"]
                    for crit, name in (("dist", "centre within one ball diameter"), ("iou", "IoU >= 0.5")):
                        rr = dict(r, tp=r[f"tp_{crit}"], fp=r[f"fp_{crit}"], fn=r[f"fn_{crit}"])
                        add_detection(run_id, scope, sid, "ball", r["method"], name, rr, r[f"precision_{crit}"],
                                      r[f"recall_{crit}"], r[f"f1_{crit}"],
                                      r.get("ap50_candidates") if crit == "iou" else None)

        # model comparison
        if (mdir / "model_comparison.csv").is_file():
            run_id = _run(conn, "model_comparison", None)
            for r in read_csv(mdir / "model_comparison.csv"):
                add_detection(run_id, f"model comparison: {r['dataset']} subset", None, r["target"], r["model"],
                              r["criterion"], {}, r["precision"], r["recall"], r["f1"], r["ap50"])

        conn.commit()
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("experiment_runs", "datasets", "sequences", "tracking_metrics", "detection_metrics",
                            "track_summaries")}
    finally:
        conn.close()
    log.info("SQLite database written: %s  %s", db_path, counts)
    return db_path
