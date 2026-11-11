"""Runtime benchmark: time every stage of the application on one sequence.

Run this on an otherwise idle machine. The detection cache is filled while
other work may be going on, so its timings can be inflated; this stage
re-measures the detector and all other stages in one clean pass and its
detector time is what the tracker comparison uses for end-to-end FPS.

Output: results/metrics/runtime_benchmark.json, results/metrics/runtime_stages.csv
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from ..detection.base import Detections
from ..detection.filters import filter_balls, filter_players
from ..tracking.ball_tracker import BallTracker
from ..tracking.factory import ALGORITHMS, TRACKER_LABELS, build_mot_tracker
from ..tracking.types import PLAYER, TrackedObject
from ..video_io import VideoWriter
from ..visualization import Visualizer
from .common import read_json, system_info, write_csv, write_json
from .manifest import Manifest
from .players import DEFAULT_PARAMS

log = logging.getLogger(__name__)

FRAMES = 150
WARMUP = 5


def _ms(samples: list[float]) -> dict:
    a = np.asarray(samples[WARMUP:] if len(samples) > WARMUP else samples) * 1000.0
    return {"mean_ms": round(float(a.mean()), 4), "median_ms": round(float(np.median(a)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4)}


def benchmark_file(manifest: Manifest) -> Path:
    return manifest.results_dir / "metrics" / "runtime_benchmark.json"


def benchmarked_detector_ms(manifest: Manifest) -> float | None:
    path = benchmark_file(manifest)
    if path.is_file():
        return float(read_json(path)["stages"]["detection"]["median_ms"])
    return None


def run_runtime_benchmark(manifest: Manifest) -> None:
    import supervision as sv

    from ..pipeline import build_yolo_detectors

    seqs = manifest.sportsmot("test") or manifest.uvy_track()
    if not seqs:
        log.error("No sequence available for the runtime benchmark.")
        return
    seq = seqs[0]
    cfg = manifest.app_config()
    sel_path = manifest.results_dir / "metrics" / "player_selected_params.json"
    selected = read_json(sel_path)["selected"] if sel_path.is_file() else {a: dict(DEFAULT_PARAMS) for a in ALGORITHMS}

    player_det, ball_det = build_yolo_detectors(cfg)
    shape = (seq.info.height, seq.info.width)
    t_read, t_det, frames, dets_p, dets_b = [], [], [], [], []
    it = seq.frames()
    while len(frames) < FRAMES:
        t0 = time.perf_counter()
        try:
            _, img = next(it)
        except StopIteration:
            break
        t_read.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        p = player_det.detect(img)
        b = ball_det.detect(img)
        t_det.append(time.perf_counter() - t0)
        frames.append(img)
        dets_p.append(p)
        dets_b.append(b)
    n = len(frames)

    stages = {"frame reading (JPEG decode)": _ms(t_read), "detection": _ms(t_det)}

    # player filter + tracker, per algorithm, on the detections just computed
    tracker_ms = {}
    tracked_for_draw: list[list[TrackedObject]] = []
    for algo in ALGORITHMS:
        tracker = build_mot_tracker(algo, seq.info.fps, **selected[algo])
        samples = []
        for i in range(n):
            t0 = time.perf_counter()
            d = filter_players(dets_p[i], shape, cfg.players.filter)
            sv_d = (sv.Detections(xyxy=d.xyxy, confidence=d.confidence, class_id=np.zeros(len(d), dtype=int))
                    if len(d) else sv.Detections.empty())
            res = tracker.update(sv_d)
            samples.append(time.perf_counter() - t0)
            if algo == "bytetrack":
                objs = []
                if res.tracker_id is not None:
                    for k, tid in enumerate(res.tracker_id):
                        if tid >= 0:
                            objs.append(TrackedObject(PLAYER, int(tid), np.asarray(res.xyxy[k], np.float32), "person",
                                                      float(res.confidence[k])))
                tracked_for_draw.append(objs)
        tracker_ms[algo] = _ms(samples)
        stages[f"player filter + {TRACKER_LABELS[algo].split(' + ')[1]}"] = tracker_ms[algo]

    # ball filter + Kalman tracker
    ball_tracker = BallTracker(cfg.ball.tracker)
    samples, balls, trajs = [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        kept = filter_balls(dets_b[i] if len(dets_b[i]) else Detections.empty(), cfg.ball.filter)
        obj = ball_tracker.update(i, kept, shape)
        samples.append(time.perf_counter() - t0)
        balls.append(obj)
        trajs.append(list(ball_tracker.trajectory) if obj is not None else [])
    stages["ball filter + Kalman tracker"] = _ms(samples)

    # drawing and video encoding
    viz = Visualizer(cfg.visualization)
    t_draw, t_write = [], []
    with tempfile.TemporaryDirectory() as tmp:
        with VideoWriter(Path(tmp) / "bench.mp4", seq.info.fps, (seq.info.width, seq.info.height), cfg.io.fourcc) as w:
            for i in range(n):
                t0 = time.perf_counter()
                out = viz.draw(frames[i], i, i / seq.info.fps, tracked_for_draw[i], balls[i], trajs[i])
                t_draw.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                w.write(out)
                t_write.append(time.perf_counter() - t0)
    stages["drawing annotations"] = _ms(t_draw)
    stages["video encoding (mp4v)"] = _ms(t_write)

    common = sum(stages[k]["median_ms"] for k in ("frame reading (JPEG decode)", "detection",
                                                  "ball filter + Kalman tracker", "drawing annotations",
                                                  "video encoding (mp4v)"))
    totals = {a: round(common + tracker_ms[a]["median_ms"], 3) for a in ALGORITHMS}
    payload = {
        "sequence": f"{seq.info.dataset}/{seq.info.name}", "frames": n, "warmup_frames_excluded": WARMUP,
        "resolution": f"{seq.info.width}x{seq.info.height}", "detector": manifest.detector_tag(),
        "system": system_info(cfg.players.detector.device), "tracker_params": selected, "stages": stages,
        "full_pipeline_ms_per_frame": totals,
        "full_pipeline_fps": {a: round(1000.0 / t, 3) for a, t in totals.items()},
    }
    write_json(benchmark_file(manifest), payload)
    write_csv(manifest.results_dir / "metrics" / "runtime_stages.csv",
              [{"stage": k, **v} for k, v in stages.items()])
    log.info("Runtime benchmark on %s (%d frames): detection %.0f ms, full pipeline %s FPS",
             payload["sequence"], n, stages["detection"]["median_ms"], payload["full_pipeline_fps"])
