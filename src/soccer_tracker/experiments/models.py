"""Optional detector comparison: YOLOv8n vs YOLO11n.

Both models are COCO-pretrained "nano" detectors run with identical settings
(image size, thresholds, classes) on the same frames: one SportsMOT sequence
for players and two InFactory clips for the ball. Speed is measured separately by
alternating the two models frame by frame, so both see the same machine state.

Output: results/metrics/model_comparison.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..evaluation import average_precision, evaluate_ball, pool
from .ball import _merge, run_ball_methods
from .common import write_csv
from .detections import get_detections
from .manifest import Manifest
from .players import player_detection_metrics

log = logging.getLogger(__name__)


TIMING_FRAMES = 40


def interleaved_timing(manifest: Manifest, weights_list: list[str], seq) -> dict[str, float]:
    """Median detector time per frame for each model, measured frame by frame in alternation.

    Timing the models in separate passes is unreliable on a laptop: a change of
    power source or thermal state between passes can alter speed by a factor of
    three. Alternating the models on every frame exposes them to the same
    conditions. The order is swapped on every frame and the first frames are
    discarded as warm-up.
    """
    import time

    from ..pipeline import build_yolo_detectors

    if seq is None:
        return {}
    dets = {w: build_yolo_detectors(manifest.app_config(w)) for w in weights_list}
    samples: dict[str, list[float]] = {w: [] for w in weights_list}
    for f, img in seq.frames():
        order = weights_list if f % 2 else weights_list[::-1]
        for w in order:
            frame = img.copy()  # a new object, so the shared-pass cache cannot be reused
            t0 = time.perf_counter()
            dets[w][0].detect(frame)
            dets[w][1].detect(frame)
            samples[w].append(time.perf_counter() - t0)
        if f >= TIMING_FRAMES + 3:
            break
    out = {w: round(1000.0 * float(np.median(v[3:])), 2) for w, v in samples.items()}
    log.info("interleaved detector timing over %d frames: %s", TIMING_FRAMES, out)
    return out


def run_model_comparison(manifest: Manifest) -> None:
    from ..pipeline import build_yolo_detectors

    sec = manifest.raw.get("model_comparison") or {}
    weights_list = sec.get("weights") or []
    if len(weights_list) < 2:
        log.warning("model_comparison needs at least two weights files in the manifest; skipping")
        return
    from ..datasets import SportsMOTSequence

    sm = manifest.raw.get("sportsmot") or {}
    player_seqs = []
    for name in sec.get("sportsmot") or []:
        try:
            player_seqs.append(SportsMOTSequence(sm["root"], name, split=sm.get("split", "val"), role="test",
                                                 max_frames=sec.get("max_frames")))
        except Exception as exc:  # dataset not downloaded
            manifest.skipped.append(f"model comparison, SportsMOT/{name}: {exc}")
    ball_seqs = manifest.infactory(only=sec.get("infactory") or [])
    timing = interleaved_timing(manifest, weights_list, player_seqs[0] if player_seqs else None)
    rows = []
    for weights in weights_list:
        cfg = manifest.app_config(weights)
        tag = manifest.detector_tag(weights) + "_cmp"
        model = Path(weights).stem.replace("yolov", "YOLOv").replace("yolo1", "YOLO1")
        detectors = build_yolo_detectors(cfg)

        p_caches = [(s, get_detections(s, cfg, manifest.results_dir, tag, detectors)) for s in player_seqs]
        b_caches = [(s, get_detections(s, cfg, manifest.results_dir, tag, detectors)) for s in ball_seqs]

        if p_caches:
            ms = [player_detection_metrics(s, c, cfg) for s, c in p_caches]
            tp, fp, fn = (sum(m[k] for m in ms) for k in ("tp", "fp", "fn"))
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            det_ms = timing.get(weights) or float(np.mean([c.detector_time()["median_ms"] for _, c in p_caches]))
            rows.append({"model": model, "target": "player", "dataset": "SportsMOT",
                         "sequences": " ".join(s.info.name for s, _ in p_caches),
                         "frames": sum(c.num_frames for _, c in p_caches),
                         "precision": round(100 * p, 3), "recall": round(100 * r, 3),
                         "f1": round(100 * 2 * p * r / max(p + r, 1e-9), 3),
                         "ap50": round(float(np.mean([m["ap50"] for m in ms])), 3),
                         "detector_ms_per_frame": round(det_ms, 2),
                         "criterion": "IoU >= 0.5, confidence >= 0.5 (AP over all confidences)"})
        if b_caches:
            results, items = [], []
            for s, c in b_caches:
                outputs, candidates, _ = run_ball_methods(c, s, cfg)
                gt = s.ball_gt()
                results.append(evaluate_ball(gt, outputs["detection_only"], s.info.num_frames))
                items.append((s, gt, candidates, set(range(1, s.info.num_frames + 1))))
            m = pool(results)
            ap = average_precision(*_merge(items), 0.5)["ap"]
            det_ms = timing.get(weights) or float(np.mean([c.detector_time()["median_ms"] for _, c in b_caches]))
            rows.append({"model": model, "target": "ball", "dataset": "InFactory",
                         "sequences": " ".join(s.info.name for s, _ in b_caches),
                         "frames": sum(c.num_frames for _, c in b_caches),
                         "precision": round(100 * m["precision_dist"], 3), "recall": round(100 * m["recall_dist"], 3),
                         "f1": round(100 * m["f1_dist"], 3), "ap50": round(100 * ap, 3),
                         "detector_ms_per_frame": round(det_ms, 2),
                         "criterion": "detection only, centre within one ball diameter (AP at IoU >= 0.5)"})
        log.info("model %s done", model)
    write_csv(manifest.results_dir / "metrics" / "model_comparison.csv", rows)
    for r in rows:
        log.info("%-8s %-6s P %.1f R %.1f F1 %.1f AP50 %.1f  %.0f ms/frame", r["model"], r["target"], r["precision"],
                 r["recall"], r["f1"], r["ap50"], r["detector_ms_per_frame"])
