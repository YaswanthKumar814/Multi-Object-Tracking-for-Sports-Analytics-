"""Detection metrics: precision, recall, F1 and average precision.

CUSTOM EVALUATOR. This is this project's own implementation of the standard
definitions, validated by unit tests in ``tests/test_evaluation.py``. It is
not the official evaluation code of any benchmark, so its numbers must not be
presented as official benchmark scores.

Definitions (single class, per-frame matching):

* A prediction matches a ground-truth box when their IoU is at least
  ``iou_threshold``. Each ground-truth box can be matched at most once.
* TP = matched predictions, FP = unmatched predictions, FN = unmatched
  ground-truth boxes.
* precision = TP / (TP + FP), recall = TP / (TP + FN),
  F1 = 2 * precision * recall / (precision + recall).
* AP: predictions from all frames are sorted by confidence and matched greedily
  (highest confidence first, as in PASCAL VOC / COCO). AP is the area under the
  precision-recall curve after making precision monotonically non-increasing
  (all-point interpolation). With one class, mAP@0.5 equals AP@0.5.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from ..detection.base import box_iou


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1}


def match_frame(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one matching that maximises total IoU (Hungarian algorithm).

    Returns ``(gt_index, pred_index)`` pairs with IoU >= ``iou_threshold``.
    """
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    if len(gt) == 0 or len(pred) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    iou = box_iou(gt, pred)
    cost = np.where(iou >= iou_threshold, 1.0 - iou, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]


def detection_counts(
    gt_by_frame: Mapping[int, np.ndarray],
    pred_by_frame: Mapping[int, np.ndarray],
    frames: Optional[set[int]] = None,
    iou_threshold: float = 0.5,
) -> dict:
    """Precision / recall / F1 pooled over frames (micro-average).

    ``frames`` limits scoring to those frames; by default every frame that has
    ground truth or predictions is scored.
    """
    if frames is None:
        frames = set(gt_by_frame) | set(pred_by_frame)
    tp = fp = fn = 0
    for f in frames:
        gt = np.asarray(gt_by_frame.get(f, np.zeros((0, 4)))).reshape(-1, 4)
        pred = np.asarray(pred_by_frame.get(f, np.zeros((0, 4)))).reshape(-1, 4)
        m = len(match_frame(gt, pred, iou_threshold))
        tp += m
        fp += len(pred) - m
        fn += len(gt) - m
    return prf(tp, fp, fn)


def average_precision(
    gt_by_frame: Mapping[int, np.ndarray],
    pred_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    frames: Optional[set[int]] = None,
    iou_threshold: float = 0.5,
) -> dict:
    """AP at one IoU threshold.

    ``pred_by_frame`` maps a frame to ``(xyxy (K, 4), confidence (K,))``.
    Returns ``{"ap": float, "precision": array, "recall": array, "num_gt": int}``.
    """
    if frames is None:
        frames = set(gt_by_frame) | set(pred_by_frame)
    num_gt = sum(len(np.asarray(gt_by_frame[f]).reshape(-1, 4)) for f in frames if f in gt_by_frame)

    scored: list[tuple[float, int, int]] = []  # confidence, frame, index
    for f in frames:
        if f in pred_by_frame:
            _, conf = pred_by_frame[f]
            scored.extend((float(c), f, i) for i, c in enumerate(np.asarray(conf).reshape(-1)))
    scored.sort(key=lambda t: -t[0])

    used: dict[int, np.ndarray] = {}
    hits = np.zeros(len(scored), dtype=bool)
    for k, (_, f, i) in enumerate(scored):
        gt = np.asarray(gt_by_frame.get(f, np.zeros((0, 4)))).reshape(-1, 4)
        if len(gt) == 0:
            continue
        taken = used.setdefault(f, np.zeros(len(gt), dtype=bool))
        box = np.asarray(pred_by_frame[f][0]).reshape(-1, 4)[i]
        iou = box_iou(box[None], gt)[0]
        iou[taken] = -1.0
        j = int(np.argmax(iou))
        if iou[j] >= iou_threshold:
            taken[j] = True
            hits[k] = True

    if num_gt == 0 or len(scored) == 0:
        return {"ap": 0.0, "precision": np.zeros(0), "recall": np.zeros(0), "num_gt": int(num_gt)}
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    recall = tp / num_gt
    precision = tp / (tp + fp)
    # all-point interpolation: monotone precision envelope, then integrate over recall
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "precision": precision, "recall": recall, "num_gt": int(num_gt)}
