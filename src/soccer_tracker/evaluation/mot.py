"""Multi-object tracking metrics (HOTA, CLEAR-MOT, Identity).

The metric computation itself is **not** implemented here. It is delegated to
``trackers.eval`` from Roboflow's ``trackers`` package, an established
re-implementation designed to match the matching logic and formulas of
TrackEval (Luiten & Hoffhues), the reference evaluator used by MOTChallenge
and SportsMOT. This module only converts between this project's arrays and
MOT Challenge text files and flattens the results.

Numbers produced here are computed on a small subset of sequences with a
generic COCO detector, so they are not comparable to leaderboard entries and
are not official benchmark scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Metrics reported everywhere, in presentation order. Ratios are stored as percentages.
RATIO_METRICS = ["HOTA", "DetA", "AssA", "OWTA", "MOTA", "IDF1", "MOTP", "CLR_Pr", "CLR_Re", "IDP", "IDR",
                 "DetPr", "DetRe", "AssPr", "AssRe", "LocA"]
COUNT_METRICS = ["IDSW", "CLR_FP", "CLR_FN", "CLR_TP", "Frag", "MT", "PT", "ML"]


def write_mot_file(path: Path, rows: np.ndarray, ground_truth: bool) -> None:
    """Write ``[frame, id, x, y, w, h(, conf)]`` rows as a MOT Challenge text file.

    Ground truth uses ``conf=1, class=1, visibility=1`` (scored pedestrian-class
    rows in MOT conventions); tracker output uses ``conf, -1, -1, -1``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, rows.shape[1] if rows.ndim == 2 else 6)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            head = f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}"
            if ground_truth:
                fh.write(head + ",1,1,1\n")
            else:
                conf = r[6] if len(r) > 6 else 1.0
                fh.write(head + f",{conf:.4f},-1,-1,-1\n")


def evaluate_mot(gt_path: Path, pred_path: Path):
    """Run trackers.eval on one sequence. Returns its ``SequenceResult``."""
    from trackers.eval.evaluate import evaluate_mot_sequence

    return evaluate_mot_sequence(gt_path, pred_path, metrics=["CLEAR", "HOTA", "Identity"], threshold=0.5)


def aggregate(results: list) -> Any:
    """Combine sequences the way TrackEval does (sum counts, recompute ratios)."""
    from trackers.eval import (
        aggregate_clear_metrics,
        aggregate_hota_metrics,
        aggregate_identity_metrics,
    )
    from trackers.eval.results import CLEARMetrics, HOTAMetrics, IdentityMetrics, SequenceResult

    return SequenceResult(
        sequence="COMBINED",
        CLEAR=CLEARMetrics.from_dict(aggregate_clear_metrics([r.CLEAR.to_dict() for r in results])),
        HOTA=HOTAMetrics.from_dict(aggregate_hota_metrics(
            [r.HOTA.to_dict(include_arrays=True, arrays_as_list=False) for r in results])),
        Identity=IdentityMetrics.from_dict(aggregate_identity_metrics([r.Identity.to_dict() for r in results])),
    )


def flatten(result) -> dict:
    """SequenceResult -> flat dict; ratio metrics as percentages, plus F1 from CLEAR precision/recall."""
    src = {}
    src.update(result.CLEAR.to_dict())
    src.update(result.HOTA.to_dict())
    src.update(result.Identity.to_dict())
    out: dict = {}
    for k in RATIO_METRICS:
        out[k] = round(100.0 * float(src[k]), 3)
    for k in COUNT_METRICS:
        out[k] = int(src[k])
    p, r = float(src["CLR_Pr"]), float(src["CLR_Re"])
    out["CLR_F1"] = round(100.0 * (2 * p * r / (p + r) if p + r else 0.0), 3)
    return out
