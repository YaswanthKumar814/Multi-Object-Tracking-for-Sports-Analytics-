"""Ball experiments: detection only vs the Kalman ball tracker.

Three outputs are compared on identical cached ball candidates:

* ``detection_only``   - the most confident ball candidate of each frame, with
                         no temporal reasoning (baseline);
* ``kalman_detected``  - positions the Kalman tracker reports from a detection
                         (ablation: shows the effect of gating / confirmation);
* ``kalman``           - the full proposed ball tracker: detected positions plus
                         positions *predicted* through short detection gaps.

No ball parameter is tuned on any evaluated sequence: the tracker runs with the
project's default configuration.

Outputs (results/metrics/): ball_per_sequence.csv, ball_summary.csv, ball.json
and one per-frame trace per sequence under results/metrics/ball_traces/.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..config import AppConfig
from ..datasets import Sequence
from ..detection.filters import filter_balls
from ..evaluation import BallOutput, average_precision, evaluate_ball, pool
from ..evaluation.ball import frame_match
from ..tracking.ball_tracker import BallTracker
from .common import system_info, write_csv, write_json
from .detections import DetectionCache, get_detections
from .manifest import Manifest

log = logging.getLogger(__name__)

METHODS = {
    "detection_only": "Detection only (YOLO11n)",
    "kalman_detected": "Kalman tracker, detected positions only",
    "kalman": "Kalman tracker: detected + predicted (proposed)",
}
RATIO_KEYS = ["precision_iou", "recall_iou", "f1_iou", "precision_dist", "recall_dist", "f1_dist", "coverage",
              "visible_coverage", "precision_detected_iou", "precision_detected_dist", "precision_predicted_iou",
              "precision_predicted_dist"]


def run_ball_methods(cache: DetectionCache, seq: Sequence, cfg: AppConfig):
    """Replay cached ball candidates. Returns (outputs per method, candidates per frame, tracker ms/frame)."""
    shape = (seq.info.height, seq.info.width)
    tracker = BallTracker(cfg.ball.tracker)
    outputs: dict[str, dict[int, BallOutput]] = {m: {} for m in METHODS}
    candidates: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    elapsed = 0.0
    for f in range(1, cache.num_frames + 1):
        cand = cache.balls(f)
        if len(cand):
            cand = cand.subset(cand.confidence >= cfg.ball.detector.confidence)
        cand = filter_balls(cand, cfg.ball.filter)
        candidates[f] = (cand.xyxy.copy(), cand.confidence.copy())
        if len(cand):
            i = int(np.argmax(cand.confidence))
            outputs["detection_only"][f] = BallOutput(cand.xyxy[i].astype(np.float64), "detected",
                                                      float(cand.confidence[i]), track_id=None)
        t0 = time.perf_counter()
        obj = tracker.update(f, cand, shape)
        elapsed += time.perf_counter() - t0
        if obj is not None:
            out = BallOutput(obj.xyxy.astype(np.float64), obj.status, obj.confidence, obj.track_id)
            outputs["kalman"][f] = out
            if obj.status == "detected":
                outputs["kalman_detected"][f] = out
    return outputs, candidates, 1000.0 * elapsed / max(cache.num_frames, 1)


def _percent(metrics: dict) -> dict:
    out = dict(metrics)
    for k in RATIO_KEYS:
        if out.get(k) is not None:
            out[k] = round(100.0 * out[k], 3)
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 3)
    return out


def _trace(seq: Sequence, gt: dict, labelled, outputs: dict, candidates: dict) -> list[dict]:
    rows = []
    for f in range(1, seq.info.num_frames + 1):
        g = gt.get(f)
        det = outputs["detection_only"].get(f)
        trk = outputs["kalman"].get(f)
        row = {
            "frame": f, "labelled": labelled is None or f in labelled, "ball_visible": g is not None,
            "gt_cx": round(float((g[0, 0] + g[0, 2]) / 2), 2) if g is not None else None,
            "gt_cy": round(float((g[0, 1] + g[0, 3]) / 2), 2) if g is not None else None,
            "num_candidates": len(candidates[f][0]),
            "det_cx": round(float(det.center[0]), 2) if det else None,
            "det_cy": round(float(det.center[1]), 2) if det else None,
            "det_conf": round(det.confidence, 4) if det else None,
            "det_correct": bool(frame_match(g, det)["dist"]) if det and g is not None else (False if det else None),
            "trk_cx": round(float(trk.center[0]), 2) if trk else None,
            "trk_cy": round(float(trk.center[1]), 2) if trk else None,
            "trk_status": trk.status if trk else "lost",
            "trk_id": trk.track_id if trk else None,
            "trk_correct": bool(frame_match(g, trk)["dist"]) if trk and g is not None else (False if trk else None),
        }
        rows.append(row)
    return rows


def run_ball_experiments(manifest: Manifest) -> None:
    cfg = manifest.app_config()
    tag = manifest.detector_tag()
    out_dir = manifest.results_dir / "metrics"
    per_seq: list[dict] = []
    raw: dict[tuple[str, str], list[dict]] = {}  # (dataset, method) -> unrounded per-sequence results
    ap_inputs: dict[str, list] = {}

    for seq in manifest.ball_sequences():
        cache = get_detections(seq, cfg, manifest.results_dir, tag, compute=False)
        if cache is None:
            manifest.skipped.append(f"{seq.info.dataset}/{seq.info.name}: no cached detections (run 'detect')")
            continue
        gt = seq.ball_gt()
        labelled = seq.ball_labelled_frames()
        gt_ids = seq.ball_track_ids() if seq.has_ball_ids else None
        outputs, candidates, tracker_ms = run_ball_methods(cache, seq, cfg)

        frames = set(range(1, seq.info.num_frames + 1)) if labelled is None else set(labelled)
        ap = average_precision(gt, candidates, frames, 0.5)["ap"]
        ap_inputs.setdefault(seq.info.dataset, []).append((seq, gt, candidates, frames))

        results = {}
        for method, label in METHODS.items():
            m = evaluate_ball(gt, outputs[method], seq.info.num_frames, labelled,
                              gt_ids if method != "detection_only" else None)
            results[method] = m
            raw.setdefault((seq.info.dataset, method), []).append(m)
            row = {"dataset": seq.info.dataset, "sequence": seq.info.name, "method": method, "method_label": label,
                   **_percent(m)}
            row["ap50_candidates"] = round(100.0 * ap, 3) if method == "detection_only" else None
            row["tracker_ms_per_frame"] = round(tracker_ms, 4) if method != "detection_only" else 0.0
            row["detector_ms_per_frame"] = round(cache.detector_time()["median_ms"], 2)
            per_seq.append(row)

        # How many visible-ball frames does the Kalman tracker add or lose vs detection only?
        trace = _trace(seq, gt, labelled, outputs, candidates)
        write_csv(out_dir / "ball_traces" / f"{seq.info.key}.csv", trace)
        vis = [r for r in trace if r["ball_visible"] and r["labelled"]]
        gained = sum(1 for r in vis if r["trk_correct"] and not r["det_correct"])
        lost = sum(1 for r in vis if r["det_correct"] and not r["trk_correct"])
        for row in per_seq[-len(METHODS):]:
            if row["method"] == "kalman":
                row["visible_frames_gained_vs_detection_only"] = gained
                row["visible_frames_lost_vs_detection_only"] = lost
        log.info("%-10s %-10s visible %4d | det-only R %.1f P %.1f | kalman R %.1f P %.1f (pred %d frames) | AP50 %.1f",
                 seq.info.dataset, seq.info.name, results["kalman"]["frames_ball_visible"],
                 100 * results["detection_only"]["recall_dist"], 100 * results["detection_only"]["precision_dist"],
                 100 * results["kalman"]["recall_dist"], 100 * results["kalman"]["precision_dist"],
                 results["kalman"]["frames_predicted"], 100 * ap)

    if not per_seq:
        log.error("No ball sequences with cached detections; nothing to do.")
        return

    # Pooled (micro-averaged) summary per dataset and over everything.
    summary: list[dict] = []
    datasets = sorted({d for d, _ in raw})
    for group in datasets + ["All ball sequences"]:
        for method, label in METHODS.items():
            parts = [m for (d, meth), ms in raw.items() if meth == method and (group == "All ball sequences" or d == group)
                     for m in ms]
            row = {"group": group, "method": method, "method_label": label, "sequences": len(parts),
                   **_percent(pool(parts))}
            if method == "detection_only":
                items = [it for d, its in ap_inputs.items() if group == "All ball sequences" or d == group for it in its]
                row["ap50_candidates"] = round(100.0 * _pooled_ap(items), 3)
            if method == "kalman":
                rows = [r for r in per_seq if r["method"] == "kalman"
                        and (group == "All ball sequences" or r["dataset"] == group)]
                row["visible_frames_gained_vs_detection_only"] = sum(
                    r.get("visible_frames_gained_vs_detection_only", 0) for r in rows)
                row["visible_frames_lost_vs_detection_only"] = sum(
                    r.get("visible_frames_lost_vs_detection_only", 0) for r in rows)
            summary.append(row)

    # Precision-recall curve of the raw ball candidates, per dataset (for the PR plot).
    curves = {}
    for d, items in ap_inputs.items():
        gt_all, pred_all, frames_all = _merge(items)
        ap = average_precision(gt_all, pred_all, frames_all, 0.5)
        step = max(1, len(ap["recall"]) // 400)
        curves[d] = {"ap50": round(100.0 * ap["ap"], 3), "num_gt": ap["num_gt"],
                     "recall": [round(100.0 * float(v), 3) for v in ap["recall"][::step]],
                     "precision": [round(100.0 * float(v), 3) for v in ap["precision"][::step]]}
    write_json(out_dir / "ball_pr_curves.json", curves)

    write_csv(out_dir / "ball_per_sequence.csv", per_seq)
    write_csv(out_dir / "ball_summary.csv", summary)
    write_json(out_dir / "ball.json", {
        "evaluator": "custom (soccer_tracker.evaluation.ball / .detection); not an official benchmark evaluator",
        "criteria": {"iou": "IoU >= 0.5 with a ground-truth ball box",
                     "dist": "centre within one ball diameter (max(gt w, gt h)) of the ground-truth centre"},
        "identity_metrics": "id_switches reported only for UVY-Track (persistent ball ids); never for InFactory",
        "detector": tag, "ball_tracker_config": cfg.ball.tracker.__dict__, "ball_filter_config": cfg.ball.filter.__dict__,
        "ball_detector_confidence": cfg.ball.detector.confidence,
        "system": system_info(cfg.ball.detector.device), "summary": summary, "per_sequence": per_seq,
    })
    log.info("Ball experiments written to %s", out_dir)


def _pooled_ap(items: list) -> float:
    """AP@0.5 over several sequences."""
    return average_precision(*_merge(items), 0.5)["ap"]


def _merge(items: list):
    """Concatenate sequences, offsetting frame numbers so they do not collide."""
    gt_all: dict[int, np.ndarray] = {}
    pred_all: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    frames_all: set[int] = set()
    offset = 0
    for seq, gt, cand, frames in items:
        for f, b in gt.items():
            gt_all[f + offset] = b
        for f, c in cand.items():
            pred_all[f + offset] = c
        frames_all |= {f + offset for f in frames}
        offset += seq.info.num_frames + 1
    return gt_all, pred_all, frames_all
