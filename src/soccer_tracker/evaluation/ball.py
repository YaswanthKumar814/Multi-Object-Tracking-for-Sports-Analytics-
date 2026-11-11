"""Ball detection / tracking metrics for a single-ball output per frame.

CUSTOM EVALUATOR. Written for this project and validated by unit tests in
``tests/test_evaluation.py``. It is not the official evaluation code of any
benchmark; do not present its numbers as official benchmark scores.

A ball method outputs at most one ball position per frame, with a status:
``detected`` (a detector box was used in this frame) or ``predicted`` (the
Kalman motion model filled in for a missing detection). Predictions are scored
exactly like detections but are always counted and reported separately, so a
predicted position is never passed off as a detection.

Only frames whose ball status is known ("labelled frames") are scored.

Matching criteria (both reported):

* ``iou``  - IoU between the output box and a ground-truth ball box >= 0.5.
  Standard, but very strict for a 10-20 px object: a 3 px shift can fail it.
* ``dist`` - the output centre lies within one ball diameter of the true
  centre: distance <= max(gt width, gt height). Scales with ball size.

Per-frame outcome under a criterion:

* TP - ball visible and the output matches it
* FP - an output that matches nothing (ball not visible, or wrong place)
* FN - ball visible and not matched (no output, or wrong place)

precision = TP / (TP + FP); recall = TP / (TP + FN) (this is the
"visible-ball recall"); F1 is their harmonic mean.

Other quantities:

* coverage           - labelled frames with any output / labelled frames
* visible_coverage   - visible-ball frames with any output / visible-ball frames
* centre error       - distance in pixels between output centre and nearest
                       ground-truth centre, over visible frames with an output
* continuity         - within each run of consecutive visible-ball frames, the
                       correctly tracked frames form "segments"; fewer, longer
                       segments mean a more continuous track. ``breaks`` counts
                       interruptions (segments minus runs that were tracked at all).
* id_switches        - only when ground-truth ball identities exist: number of
                       times the method's own track id changes between
                       consecutive correctly matched frames of one true ball.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from ..detection.base import box_iou
from .detection import prf

DETECTED = "detected"
PREDICTED = "predicted"
CRITERIA = ("iou", "dist")


@dataclass
class BallOutput:
    xyxy: np.ndarray  # (4,)
    status: str = DETECTED
    confidence: Optional[float] = None
    track_id: Optional[int] = None

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def _centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def frame_match(gt: np.ndarray, out: BallOutput, iou_threshold: float = 0.5) -> dict:
    """Compare one output with the ground-truth boxes of one frame."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    dists = np.linalg.norm(_centers(gt) - out.center[None], axis=1)
    j = int(np.argmin(dists))
    diameter = max(gt[j, 2] - gt[j, 0], gt[j, 3] - gt[j, 1])
    iou = float(box_iou(out.xyxy[None], gt).max())
    return {"iou": iou >= iou_threshold, "dist": bool(dists[j] <= diameter), "error_px": float(dists[j]),
            "iou_value": iou}


def _runs(frames: list[int]) -> list[list[int]]:
    """Split a sorted list of frame numbers into runs of consecutive frames."""
    runs: list[list[int]] = []
    for f in frames:
        if runs and f == runs[-1][-1] + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def evaluate_ball(
    gt_by_frame: Mapping[int, np.ndarray],
    outputs: Mapping[int, BallOutput],
    num_frames: int,
    labelled_frames: Optional[set[int]] = None,
    gt_ids: Optional[Mapping[int, int]] = None,
    iou_threshold: float = 0.5,
) -> dict:
    """Score one method on one sequence. Returns a flat dict of metrics."""
    labelled = set(range(1, num_frames + 1)) if labelled_frames is None else set(labelled_frames)
    visible = sorted(f for f in gt_by_frame if f in labelled)
    scored_out = {f: o for f, o in outputs.items() if f in labelled}

    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CRITERIA}
    by_status = {s: {"n": 0, "tp_iou": 0, "tp_dist": 0} for s in (DETECTED, PREDICTED)}
    errors: list[float] = []
    errors_by_status = {DETECTED: [], PREDICTED: []}
    errors_correct: list[float] = []  # localisation error of outputs that are correct (distance criterion)
    correct = {c: set() for c in CRITERIA}

    for f in sorted(labelled):
        out = scored_out.get(f)
        gt = gt_by_frame.get(f)
        if out is not None:
            by_status[out.status]["n"] += 1
        if gt is None:
            if out is not None:
                for c in CRITERIA:
                    counts[c]["fp"] += 1
            continue
        if out is None:
            for c in CRITERIA:
                counts[c]["fn"] += 1
            continue
        m = frame_match(gt, out, iou_threshold)
        errors.append(m["error_px"])
        errors_by_status[out.status].append(m["error_px"])
        if m["dist"]:
            errors_correct.append(m["error_px"])
        for c in CRITERIA:
            if m[c]:
                counts[c]["tp"] += 1
                correct[c].add(f)
                by_status[out.status][f"tp_{c}"] += 1
            else:  # wrong place: a false alarm and a miss at once
                counts[c]["fp"] += 1
                counts[c]["fn"] += 1

    n_lab = len(labelled)
    res: dict = {
        "frames_labelled": n_lab,
        "frames_ball_visible": len(visible),
        "frames_with_output": len(scored_out),
        "frames_detected": by_status[DETECTED]["n"],
        "frames_predicted": by_status[PREDICTED]["n"],
        "frames_lost": n_lab - len(scored_out),
        "coverage": len(scored_out) / n_lab if n_lab else 0.0,
        "visible_coverage": sum(1 for f in visible if f in scored_out) / len(visible) if visible else 0.0,
        "center_error_mean_px": float(np.mean(errors)) if errors else None,
        "center_error_median_px": float(np.median(errors)) if errors else None,
        "center_error_detected_px": float(np.mean(errors_by_status[DETECTED])) if errors_by_status[DETECTED] else None,
        "center_error_predicted_px": float(np.mean(errors_by_status[PREDICTED])) if errors_by_status[PREDICTED] else None,
    }
    for c in CRITERIA:
        m = prf(**counts[c])
        for k, v in m.items():
            res[f"{k}_{c}"] = v
    for s in (DETECTED, PREDICTED):
        n = by_status[s]["n"]
        res[f"precision_{s}_iou"] = by_status[s]["tp_iou"] / n if n else None
        res[f"precision_{s}_dist"] = by_status[s]["tp_dist"] / n if n else None
        res[f"tp_{s}_iou"] = by_status[s]["tp_iou"]
        res[f"tp_{s}_dist"] = by_status[s]["tp_dist"]
    res["center_error_correct_px"] = float(np.mean(errors_correct)) if errors_correct else None

    # Continuity of the correctly tracked ball (distance criterion).
    visible_runs = _runs(visible)
    segments = [seg for run in visible_runs for seg in _runs([f for f in run if f in correct["dist"]])]
    tracked_runs = sum(1 for run in visible_runs if any(f in correct["dist"] for f in run))
    res.update({
        "visible_runs": len(visible_runs),
        "track_segments": len(segments),
        "track_breaks": len(segments) - tracked_runs,
        "longest_segment_frames": max((len(s) for s in segments), default=0),
        "mean_segment_frames": float(np.mean([len(s) for s in segments])) if segments else 0.0,
    })

    # Identity consistency, only with ground-truth ball identities.
    if gt_ids is not None:
        switches = 0
        last: dict[int, int] = {}
        used_ids: set[int] = set()
        for f in sorted(correct["dist"]):
            out = scored_out[f]
            if out.track_id is None or f not in gt_ids:
                continue
            g = gt_ids[f]
            used_ids.add(out.track_id)
            if g in last and last[g] != out.track_id:
                switches += 1
            last[g] = out.track_id
        res["id_switches"] = switches
        res["predicted_ids_used"] = len(used_ids)
        res["gt_ball_tracks"] = len(set(gt_ids[f] for f in visible if f in gt_ids))
    else:
        res["id_switches"] = None
        res["predicted_ids_used"] = None
        res["gt_ball_tracks"] = None
    return res


def pool(results: list[dict]) -> dict:
    """Micro-average several sequences: sum the counts, then recompute the ratios."""
    if not results:
        return {}
    out: dict = {}
    sum_keys = ["frames_labelled", "frames_ball_visible", "frames_with_output", "frames_detected",
                "frames_predicted", "frames_lost", "visible_runs", "track_segments", "track_breaks"]
    for k in sum_keys:
        out[k] = int(sum(r[k] for r in results))
    out["coverage"] = out["frames_with_output"] / out["frames_labelled"] if out["frames_labelled"] else 0.0
    vis_out = sum(r["visible_coverage"] * r["frames_ball_visible"] for r in results)
    out["visible_coverage"] = vis_out / out["frames_ball_visible"] if out["frames_ball_visible"] else 0.0
    for c in CRITERIA:
        m = prf(sum(r[f"tp_{c}"] for r in results), sum(r[f"fp_{c}"] for r in results),
                sum(r[f"fn_{c}"] for r in results))
        for k, v in m.items():
            out[f"{k}_{c}"] = v
    # centre error: weight each sequence by its number of visible frames with an output
    w = [r["visible_coverage"] * r["frames_ball_visible"] for r in results]
    # precision of detected and of predicted positions, pooled from their own counts
    for s, n_key in ((DETECTED, "frames_detected"), (PREDICTED, "frames_predicted")):
        for c in CRITERIA:
            out[f"tp_{s}_{c}"] = int(sum(r[f"tp_{s}_{c}"] for r in results))
            out[f"precision_{s}_{c}"] = out[f"tp_{s}_{c}"] / out[n_key] if out[n_key] else None
    # localisation error of correct outputs: weight each sequence by its number of correct outputs
    cvals = [(r["center_error_correct_px"], r["tp_dist"]) for r in results
             if r.get("center_error_correct_px") is not None and r["tp_dist"] > 0]
    out["center_error_correct_px"] = (float(sum(v * n for v, n in cvals) / sum(n for _, n in cvals))
                                      if cvals else None)
    for key in ("center_error_mean_px",):
        vals = [(r[key], wi) for r, wi in zip(results, w) if r[key] is not None and wi > 0]
        out[key] = float(sum(v * wi for v, wi in vals) / sum(wi for _, wi in vals)) if vals else None
    out["longest_segment_frames"] = max(r["longest_segment_frames"] for r in results)
    out["mean_segment_frames"] = (
        float(sum(r["mean_segment_frames"] * r["track_segments"] for r in results) / out["track_segments"])
        if out["track_segments"] else 0.0
    )
    ids = [r["id_switches"] for r in results if r.get("id_switches") is not None]
    out["id_switches"] = int(sum(ids)) if ids else None
    return out
