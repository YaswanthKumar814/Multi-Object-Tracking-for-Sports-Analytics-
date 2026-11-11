"""Player-tracking baseline comparison: SORT vs OC-SORT vs ByteTrack.

Fairness rules implemented here:

* every tracker replays the *same cached YOLO11n detections* on the same
  frames and is scored against the same ground truth with the same evaluator;
* the same geometric player filter is applied before every tracker;
* hyper-parameters are chosen per tracker with the same grid and the same
  selection rule (highest HOTA) on one SportsMOT "dev" sequence that comes from
  a different source video than every test sequence. Test sequences are never
  used for tuning. Results with untuned shared defaults are also reported.

Outputs (results/metrics/): player_tracking_per_sequence.csv,
player_tracking_summary.csv, player_tuning_grid.csv, player_selected_params.json,
player_detection.csv, plus MOT text files of every run under results/tracks/.
"""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path

import numpy as np

from ..config import AppConfig, PlayerFilterConfig
from ..datasets import Sequence
from ..detection.filters import filter_players
from ..evaluation import aggregate, average_precision, detection_counts, evaluate_mot, flatten, write_mot_file
from ..tracking.factory import ALGORITHMS, TRACKER_LABELS, accepted_params, build_mot_tracker
from .common import system_info, write_csv, write_json
from .detections import DetectionCache, get_detections
from .manifest import Manifest

log = logging.getLogger(__name__)

# Shared defaults = the values this project already used for ByteTrack before any tuning.
DEFAULT_PARAMS = {
    "lost_track_buffer": 30,
    "minimum_consecutive_frames": 3,
    "minimum_iou_threshold": 0.1,
    "track_activation_threshold": 0.5,
    "high_conf_det_threshold": 0.5,
}

# Search space used for every tracker on the dev sequence. One confidence value
# drives both confidence parameters so each tracker gets the same budget (48 runs).
GRID = {
    "lost_track_buffer": [30, 60],
    "minimum_consecutive_frames": [1, 3],
    "minimum_iou_threshold": [0.1, 0.2, 0.3],
    "confidence": [0.3, 0.4, 0.5, 0.6],
}

TIMING_REPEATS = 3


def grid_configs() -> list[dict]:
    keys = list(GRID)
    out = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        p = dict(zip(keys, values))
        conf = p.pop("confidence")
        p["track_activation_threshold"] = conf
        p["high_conf_det_threshold"] = conf
        out.append(p)
    return out


def run_tracker(algorithm: str, params: dict, cache: DetectionCache, seq: Sequence,
                filter_cfg: PlayerFilterConfig, min_confidence: float) -> tuple[np.ndarray, float]:
    """Replay cached detections through one tracker.

    Returns ``(rows [frame, id, x, y, w, h, conf], seconds spent in filtering + tracking)``.
    """
    import supervision as sv

    tracker = build_mot_tracker(algorithm, seq.info.fps, **params)
    shape = (seq.info.height, seq.info.width)
    rows: list[list[float]] = []
    elapsed = 0.0
    for f in range(1, cache.num_frames + 1):
        dets = cache.players(f)
        t0 = time.perf_counter()
        if len(dets):
            dets = dets.subset(dets.confidence >= min_confidence)
        dets = filter_players(dets, shape, filter_cfg)
        if len(dets):
            sv_dets = sv.Detections(xyxy=dets.xyxy.astype(np.float32), confidence=dets.confidence.astype(np.float32),
                                    class_id=np.zeros(len(dets), dtype=int))
        else:
            sv_dets = sv.Detections.empty()
        res = tracker.update(sv_dets)
        elapsed += time.perf_counter() - t0
        if res.tracker_id is None:
            continue
        for i, tid in enumerate(res.tracker_id):
            if tid < 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in res.xyxy[i])
            conf = float(res.confidence[i]) if res.confidence is not None else 1.0
            rows.append([f, int(tid), x1, y1, x2 - x1, y2 - y1, conf])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7), elapsed


def _gt_file(manifest: Manifest, seq: Sequence) -> Path:
    path = manifest.results_dir / "tracks" / "gt" / f"{seq.info.key}.txt"
    write_mot_file(path, seq.player_gt(), ground_truth=True)
    return path


def _evaluate(manifest: Manifest, seq: Sequence, gt_path: Path, rows: np.ndarray, subdir: str):
    pred_path = manifest.results_dir / "tracks" / subdir / f"{seq.info.key}.txt"
    if len(rows) == 0:  # trackers.eval needs at least one row; an off-screen 1-px box matches nothing
        rows = np.array([[1, 1, -10.0, -10.0, 1.0, 1.0, 0.0]])
    write_mot_file(pred_path, rows, ground_truth=False)
    return evaluate_mot(gt_path, pred_path)


def tune_on_dev(manifest: Manifest, cfg: AppConfig, caches: dict, dev: list[Sequence]) -> tuple[dict, list[dict]]:
    """Grid search per tracker on the dev sequence(s); returns the selected parameters."""
    grid_rows: list[dict] = []
    selected: dict[str, dict] = {}
    gt_paths = {s.info.key: _gt_file(manifest, s) for s in dev}
    configs = grid_configs()
    for algo in ALGORITHMS:
        seen: dict[tuple, dict] = {}
        for params in configs:
            # A tracker ignores parameters it does not have, so some grid points are duplicates.
            effective = {k: params[k] for k in accepted_params(algo)}
            key = tuple(sorted(effective.items()))
            if key not in seen:
                results = []
                for seq in dev:
                    rows, _ = run_tracker(algo, params, caches[seq.info.key], seq, cfg.players.filter,
                                          cfg.players.detector.confidence)
                    results.append(_evaluate(manifest, seq, gt_paths[seq.info.key], rows, f"tuning/{algo}"))
                seen[key] = flatten(aggregate(results) if len(results) > 1 else results[0])
            m = seen[key]
            grid_rows.append({"tracker": algo, **params, "HOTA": m["HOTA"], "MOTA": m["MOTA"], "IDF1": m["IDF1"],
                              "IDSW": m["IDSW"], "Frag": m["Frag"]})
        best = max((r for r in grid_rows if r["tracker"] == algo), key=lambda r: (r["HOTA"], r["IDF1"]))
        selected[algo] = {k: best[k] for k in DEFAULT_PARAMS}
        log.info("dev-tuned %-9s HOTA %.2f  %s", algo, best["HOTA"], selected[algo])
    return selected, grid_rows


def player_detection_metrics(seq: Sequence, cache: DetectionCache, cfg: AppConfig, threshold: float = 0.5) -> dict:
    """Detector-only quality on player boxes (custom evaluator, IoU 0.5)."""
    shape = (seq.info.height, seq.info.width)
    gt_rows = seq.player_gt()
    gt_by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(gt_rows[:, 0]).astype(int):
        r = gt_rows[gt_rows[:, 0] == f]
        gt_by_frame[int(f)] = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], axis=1)
    pred_hi: dict[int, np.ndarray] = {}
    pred_all: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f in range(1, cache.num_frames + 1):
        d = cache.players(f)
        if len(d):
            d = d.subset(d.confidence >= cfg.players.detector.confidence)
        d = filter_players(d, shape, cfg.players.filter)
        pred_all[f] = (d.xyxy, d.confidence)
        pred_hi[f] = d.xyxy[d.confidence >= threshold]
    frames = set(range(1, cache.num_frames + 1))
    m = detection_counts(gt_by_frame, pred_hi, frames, 0.5)
    ap = average_precision(gt_by_frame, pred_all, frames, 0.5)
    return {"dataset": seq.info.dataset, "sequence": seq.info.name, "role": seq.info.role, "target": "player",
            "method": "YOLO11n detector", "confidence_threshold": threshold, "iou_threshold": 0.5,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": round(100 * m["precision"], 3), "recall": round(100 * m["recall"], 3),
            "f1": round(100 * m["f1"], 3), "ap50": round(100 * ap["ap"], 3)}


def run_player_experiments(manifest: Manifest) -> None:
    cfg = manifest.app_config()
    tag = manifest.detector_tag()
    out_dir = manifest.results_dir / "metrics"
    dev = manifest.sportsmot("dev")
    tests = manifest.sportsmot("test") + manifest.uvy_track()
    sequences = dev + tests
    caches = {}
    for seq in list(sequences):
        cache = get_detections(seq, cfg, manifest.results_dir, tag, compute=False)
        if cache is None:
            manifest.skipped.append(f"{seq.info.dataset}/{seq.info.name}: no cached detections (run 'detect')")
            sequences.remove(seq)
            continue
        caches[seq.info.key] = cache
    dev = [s for s in dev if s.info.key in caches]
    tests = [s for s in tests if s.info.key in caches]
    if not tests:
        log.error("No test sequences with cached detections; nothing to do.")
        return

    # 1. choose parameters on the dev sequence only
    if dev:
        selected, grid_rows = tune_on_dev(manifest, cfg, caches, dev)
        write_csv(out_dir / "player_tuning_grid.csv", grid_rows)
    else:
        log.warning("No dev sequence available; the 'dev-tuned' parameter set falls back to the defaults.")
        selected = {a: dict(DEFAULT_PARAMS) for a in ALGORITHMS}
    param_sets = {"dev-tuned": selected, "default": {a: dict(DEFAULT_PARAMS) for a in ALGORITHMS}}
    write_json(out_dir / "player_selected_params.json", {
        "selection_rule": "highest HOTA (ties: IDF1) on the dev sequence(s)",
        "dev_sequences": [f"{s.info.dataset}/{s.info.name}" for s in dev],
        "grid": GRID, "default": DEFAULT_PARAMS, "selected": selected,
        "accepted_parameters": {a: list(accepted_params(a)) for a in ALGORITHMS},
    })

    # 2. evaluate every tracker / parameter set on every sequence
    # Detector time: one clean measurement from the runtime benchmark if it exists (identical for all
    # trackers, which share detections); otherwise the time recorded while the cache was filled.
    from .runtime import benchmarked_detector_ms

    bench_det_ms = benchmarked_detector_ms(manifest)
    per_seq: list[dict] = []
    summary: list[dict] = []
    gt_paths = {s.info.key: _gt_file(manifest, s) for s in sequences}
    for set_name, per_algo in param_sets.items():
        for algo in ALGORITHMS:
            params = per_algo[algo]
            results = {}
            timing = {}
            for seq in sequences:
                cache = caches[seq.info.key]
                runs = [run_tracker(algo, params, cache, seq, cfg.players.filter, cfg.players.detector.confidence)
                        for _ in range(TIMING_REPEATS)]
                rows = runs[0][0]
                tracker_ms = 1000.0 * float(np.median([r[1] for r in runs])) / cache.num_frames
                det_ms = bench_det_ms if bench_det_ms is not None else cache.detector_time()["median_ms"]
                res = _evaluate(manifest, seq, gt_paths[seq.info.key], rows, f"{set_name}/{algo}")
                results[seq.info.key] = res
                timing[seq.info.key] = (tracker_ms, det_ms, cache.num_frames)
                gt = seq.player_gt()
                per_seq.append({
                    "dataset": seq.info.dataset, "sequence": seq.info.name, "role": seq.info.role,
                    "tracker": algo, "tracker_label": TRACKER_LABELS[algo], "param_set": set_name,
                    "frames": cache.num_frames, "gt_tracks": int(len(np.unique(gt[:, 1]))),
                    "gt_boxes": int(len(gt)), "pred_tracks": int(len(np.unique(rows[:, 1]))) if len(rows) else 0,
                    **flatten(res),
                    "tracker_ms_per_frame": round(tracker_ms, 4), "detector_ms_per_frame": round(det_ms, 2),
                    "total_ms_per_frame": round(det_ms + tracker_ms, 2),
                    "end_to_end_fps": round(1000.0 / (det_ms + tracker_ms), 3),
                    "tracker_only_fps": round(1000.0 / tracker_ms, 1) if tracker_ms > 0 else None,
                })
            groups = {
                "SportsMOT (test)": [s for s in tests if s.info.dataset == "SportsMOT"],
                "UVY-Track (test)": [s for s in tests if s.info.dataset == "UVY-Track"],
                "All test sequences": tests,
                "SportsMOT (dev, used for tuning)": dev,
            }
            for group, seqs in groups.items():
                if not seqs:
                    continue
                agg = aggregate([results[s.info.key] for s in seqs]) if len(seqs) > 1 else results[seqs[0].info.key]
                frames = sum(timing[s.info.key][2] for s in seqs)
                tr_ms = sum(timing[s.info.key][0] * timing[s.info.key][2] for s in seqs) / frames
                det_ms = sum(timing[s.info.key][1] * timing[s.info.key][2] for s in seqs) / frames
                summary.append({
                    "group": group, "tracker": algo, "tracker_label": TRACKER_LABELS[algo], "param_set": set_name,
                    "sequences": len(seqs), "frames": frames, **flatten(agg),
                    "tracker_ms_per_frame": round(tr_ms, 4), "detector_ms_per_frame": round(det_ms, 2),
                    "total_ms_per_frame": round(det_ms + tr_ms, 2),
                    "end_to_end_fps": round(1000.0 / (det_ms + tr_ms), 3),
                    "tracker_only_fps": round(1000.0 / tr_ms, 1) if tr_ms > 0 else None,
                    **{f"param_{k}": v for k, v in params.items() if k in accepted_params(algo)},
                })
            row = next(r for r in summary if r["group"] == "All test sequences" and r["tracker"] == algo
                       and r["param_set"] == set_name)
            log.info("%-9s %-10s all-test HOTA %.2f DetA %.2f AssA %.2f MOTA %.2f IDF1 %.2f IDSW %d Frag %d",
                     set_name, algo, row["HOTA"], row["DetA"], row["AssA"], row["MOTA"], row["IDF1"], row["IDSW"],
                     row["Frag"])

    write_csv(out_dir / "player_tracking_per_sequence.csv", per_seq)
    write_csv(out_dir / "player_tracking_summary.csv", summary)
    write_json(out_dir / "player_tracking.json", {
        "evaluator": "trackers.eval (Roboflow trackers %s), TrackEval-compatible HOTA / CLEAR / Identity, IoU 0.5"
                     % __import__("importlib.metadata").metadata.version("trackers"),
        "detector": tag, "system": system_info(cfg.players.detector.device),
        "detector_time_source": "runtime benchmark (idle machine)" if bench_det_ms is not None
        else "detection cache (machine may have been busy)",
        "parameter_sets": param_sets, "summary": summary, "per_sequence": per_seq,
    })

    # 3. detector-only quality on player boxes
    det_rows = [player_detection_metrics(s, caches[s.info.key], cfg) for s in sequences]
    write_csv(out_dir / "player_detection.csv", det_rows)
    log.info("Player experiments written to %s", out_dir)
