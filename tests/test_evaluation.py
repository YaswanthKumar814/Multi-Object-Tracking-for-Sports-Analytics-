"""Validation of the evaluators with small hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_tracker.evaluation import (
    BallOutput,
    average_precision,
    detection_counts,
    evaluate_ball,
    evaluate_mot,
    flatten,
    match_frame,
    pool,
    prf,
    write_mot_file,
)


def box(cx, cy, size=10.0):
    return np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], dtype=float)


# --------------------------------------------------------------------------- #
# precision / recall / F1 / AP (custom evaluator)
# --------------------------------------------------------------------------- #
def test_prf_values_and_zero_division():
    m = prf(tp=6, fp=2, fn=4)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert prf(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_match_frame_is_one_to_one_and_respects_threshold():
    gt = np.stack([box(10, 10), box(100, 100)])
    pred = np.stack([box(100, 100), box(11, 10), box(12, 10), box(300, 300)])
    pairs = match_frame(gt, pred, 0.5)
    assert sorted(pairs) == [(0, 1), (1, 0)]  # best IoU wins; the duplicate and the far box stay unmatched
    assert match_frame(gt, np.zeros((0, 4))) == []
    assert match_frame(np.stack([box(10, 10)]), np.stack([box(18, 10)]), 0.5) == []  # IoU 0.11


def test_detection_counts_pools_frames():
    gt = {1: np.stack([box(10, 10), box(50, 50)]), 2: np.stack([box(10, 10)])}
    pred = {1: np.stack([box(10, 10)]), 2: np.stack([box(10, 10), box(200, 200)]), 3: np.stack([box(5, 5)])}
    m = detection_counts(gt, pred)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)
    # restricting the scored frames drops the false positive in frame 3
    assert detection_counts(gt, pred, frames={1, 2})["fp"] == 1


def test_average_precision_perfect_and_hand_computed():
    gt = {1: np.stack([box(10, 10)]), 2: np.stack([box(50, 50)])}
    perfect = {1: (np.stack([box(10, 10)]), np.array([0.9])), 2: (np.stack([box(50, 50)]), np.array([0.8]))}
    assert average_precision(gt, perfect)["ap"] == pytest.approx(1.0)

    # ranked: TP (0.9), FP (0.8), TP (0.7) with 2 ground-truth boxes
    # precision = [1, 1/2, 2/3], recall = [1/2, 1/2, 1] -> AP = 0.5 * 1 + 0.5 * 2/3
    mixed = {1: (np.stack([box(10, 10), box(300, 300)]), np.array([0.9, 0.8])),
             2: (np.stack([box(50, 50)]), np.array([0.7]))}
    assert average_precision(gt, mixed)["ap"] == pytest.approx(0.5 + 0.5 * 2 / 3)

    # a duplicate detection of an already matched box is a false positive
    dup = {1: (np.stack([box(10, 10), box(10, 10)]), np.array([0.9, 0.8]))}
    assert average_precision({1: np.stack([box(10, 10)])}, dup)["ap"] == pytest.approx(1.0)
    assert average_precision(gt, {})["ap"] == 0.0


# --------------------------------------------------------------------------- #
# ball evaluator (custom)
# --------------------------------------------------------------------------- #
@pytest.fixture
def ball_case():
    # 10 labelled frames; ball visible in frames 1-8 moving right 10 px per frame.
    gt = {f: box(100 + 10 * f, 50)[None] for f in range(1, 9)}
    out = {
        1: BallOutput(box(110, 50), "detected", 0.9, 0),
        2: BallOutput(box(120, 50), "detected", 0.9, 0),
        3: BallOutput(box(131, 50), "detected", 0.9, 0),  # 1 px off: still correct
        # frame 4: nothing
        5: BallOutput(box(150, 50), "predicted", None, 0),
        6: BallOutput(box(400, 300), "detected", 0.5, 1),  # wrong place
        # frames 7, 8: nothing
        9: BallOutput(box(10, 10), "detected", 0.4, 1),  # ball not visible: false alarm
    }
    return gt, out


def test_ball_counts_precision_recall(ball_case):
    gt, out = ball_case
    m = evaluate_ball(gt, out, num_frames=10)
    assert (m["tp_dist"], m["fp_dist"], m["fn_dist"]) == (4, 2, 4)
    assert m["precision_dist"] == pytest.approx(4 / 6)
    assert m["recall_dist"] == pytest.approx(4 / 8)
    assert m["tp_iou"] == 4  # the 1 px shift keeps IoU above 0.5
    assert m["frames_ball_visible"] == 8 and m["frames_labelled"] == 10
    assert m["frames_with_output"] == 6 and m["frames_lost"] == 4
    assert m["coverage"] == pytest.approx(0.6)
    assert m["visible_coverage"] == pytest.approx(5 / 8)


def test_ball_predictions_are_reported_separately(ball_case):
    gt, out = ball_case
    m = evaluate_ball(gt, out, num_frames=10)
    assert m["frames_detected"] == 5 and m["frames_predicted"] == 1
    assert m["precision_predicted_dist"] == pytest.approx(1.0)
    assert m["precision_detected_dist"] == pytest.approx(3 / 5)
    assert m["center_error_predicted_px"] == pytest.approx(0.0)


def test_ball_continuity_and_centre_error(ball_case):
    gt, out = ball_case
    m = evaluate_ball(gt, out, num_frames=10)
    # correct frames 1-3 and 5 inside one visible run -> 2 segments, 1 break
    assert m["visible_runs"] == 1 and m["track_segments"] == 2 and m["track_breaks"] == 1
    assert m["longest_segment_frames"] == 3
    # errors over visible frames with an output: 0, 0, 1, 0, |(400,300)-(160,50)|
    far = float(np.hypot(240, 250))
    assert m["center_error_mean_px"] == pytest.approx((1 + far) / 5)
    assert m["center_error_median_px"] == pytest.approx(0.0)


def test_ball_iou_criterion_is_stricter_than_distance():
    gt = {1: box(100, 100, 10)[None]}
    out = {1: BallOutput(box(106, 100, 10), "detected", 0.9)}  # 6 px shift: IoU 0.25, within one diameter
    m = evaluate_ball(gt, out, num_frames=1)
    assert m["tp_dist"] == 1 and m["tp_iou"] == 0


def test_ball_unlabelled_frames_are_not_scored():
    gt = {2: box(50, 50)[None]}
    out = {1: BallOutput(box(5, 5)), 2: BallOutput(box(50, 50)), 3: BallOutput(box(5, 5))}
    m = evaluate_ball(gt, out, num_frames=3, labelled_frames={2})
    assert (m["tp_dist"], m["fp_dist"], m["fn_dist"]) == (1, 0, 0)
    assert m["frames_labelled"] == 1


def test_ball_id_switches_only_with_ground_truth_ids(ball_case):
    gt, out = ball_case
    assert evaluate_ball(gt, out, 10)["id_switches"] is None
    out[7] = BallOutput(box(170, 50), "detected", 0.9, 2)  # same true ball, new track id
    m = evaluate_ball(gt, out, 10, gt_ids={f: 7 for f in gt})
    assert m["id_switches"] == 1 and m["predicted_ids_used"] == 2 and m["gt_ball_tracks"] == 1


def test_pool_sums_counts_then_recomputes(ball_case):
    gt, out = ball_case
    a = evaluate_ball(gt, out, 10)
    b = evaluate_ball({1: box(10, 10)[None]}, {1: BallOutput(box(10, 10))}, 1)
    p = pool([a, b])
    assert (p["tp_dist"], p["fp_dist"], p["fn_dist"]) == (5, 2, 4)
    assert p["precision_dist"] == pytest.approx(5 / 7)
    assert p["frames_labelled"] == 11 and p["coverage"] == pytest.approx(7 / 11)
    assert pool([]) == {}


# --------------------------------------------------------------------------- #
# MOT metrics (delegated to trackers.eval) - check the file conversion end to end
# --------------------------------------------------------------------------- #
def _tracks(ids_per_frame):
    """Two objects moving right; ``ids_per_frame[f] = (id_a, id_b)``."""
    rows = []
    for f, (ia, ib) in enumerate(ids_per_frame, start=1):
        rows.append([f, ia, 10 + 5 * f, 20, 20, 40, 0.9])
        rows.append([f, ib, 200 + 5 * f, 120, 20, 40, 0.9])
    return np.array(rows, dtype=float)


def test_mot_perfect_tracking_scores_100(tmp_path):
    rows = _tracks([(1, 2)] * 10)
    write_mot_file(tmp_path / "gt.txt", rows[:, :6], ground_truth=True)
    write_mot_file(tmp_path / "pred.txt", rows, ground_truth=False)
    m = flatten(evaluate_mot(tmp_path / "gt.txt", tmp_path / "pred.txt"))
    assert m["HOTA"] == pytest.approx(100.0) and m["MOTA"] == pytest.approx(100.0)
    assert m["IDF1"] == pytest.approx(100.0)
    assert (m["IDSW"], m["CLR_FP"], m["CLR_FN"], m["Frag"]) == (0, 0, 0, 0)


def test_mot_identity_switch_and_missed_detections_are_counted(tmp_path):
    gt = _tracks([(1, 2)] * 10)
    pred = _tracks([(1, 2)] * 5 + [(3, 2)] * 5)  # object A changes id half-way
    pred = pred[~((pred[:, 0] == 8) & (pred[:, 1] == 2))]  # object B missed in frame 8
    write_mot_file(tmp_path / "gt.txt", gt[:, :6], ground_truth=True)
    write_mot_file(tmp_path / "pred.txt", pred, ground_truth=False)
    m = flatten(evaluate_mot(tmp_path / "gt.txt", tmp_path / "pred.txt"))
    assert m["IDSW"] == 1 and m["CLR_FN"] == 1 and m["CLR_FP"] == 0 and m["Frag"] == 1
    assert m["MOTA"] == pytest.approx(100 * (1 - (1 + 0 + 1) / 20))
    assert m["AssA"] < 100 and m["IDF1"] < 100


def test_ball_correct_only_error_and_pooled_status_precision(ball_case):
    gt, out = ball_case
    a = evaluate_ball(gt, out, 10)
    # correct outputs are frames 1, 2, 3 (1 px off) and 5 -> mean error 0.25 px
    assert a["center_error_correct_px"] == pytest.approx(0.25)
    assert a["tp_detected_dist"] == 3 and a["tp_predicted_dist"] == 1
    b = evaluate_ball({1: box(10, 10)[None]}, {1: BallOutput(box(300, 300), "predicted")}, 1)
    p = pool([a, b])
    assert p["precision_predicted_dist"] == pytest.approx(1 / 2)  # 1 correct of 2 predicted frames
    assert p["precision_detected_dist"] == pytest.approx(3 / 5)
    assert p["center_error_correct_px"] == pytest.approx(0.25)
