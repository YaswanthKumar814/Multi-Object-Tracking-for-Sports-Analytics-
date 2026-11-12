"""Build the result tables of the report directly from the stored metrics.

    python scripts/make_report_tables.py

* writes every table as Markdown to results/tables.md;
* replaces the blocks between ``<!-- TABLE:name -->`` and ``<!-- /TABLE:name -->``
  in docs/FINAL_REPORT.md, so numbers in the report are never typed by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from soccer_tracker.experiments import load_manifest
from soccer_tracker.experiments.common import read_csv, read_json
from soccer_tracker.tracking.factory import ALGORITHMS, SHORT_LABELS, TRACKER_LABELS, accepted_params


def md(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def bold_best(rows: list[list], cols: list[int], higher: dict[int, bool]) -> list[list]:
    """Bold the best value in each given column (ties all bolded)."""
    rows = [list(r) for r in rows]
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r[c], (int, float))]
        if not vals:
            continue
        best = max(vals) if higher[c] else min(vals)
        for r in rows:
            if isinstance(r[c], (int, float)) and r[c] == best:
                r[c] = f"**{r[c]:.2f}**" if isinstance(r[c], float) else f"**{r[c]}**"
    return rows


def tracking_table(summary: list[dict], group: str, param_set: str) -> str:
    rows = []
    for a in ALGORITHMS:
        r = next((x for x in summary if x["group"] == group and x["tracker"] == a and x["param_set"] == param_set), None)
        if r is None:
            return f"_No results for {group}._"
        rows.append([TRACKER_LABELS[a], r["HOTA"], r["DetA"], r["AssA"], r["MOTA"], r["IDF1"], r["IDSW"], r["CLR_FP"],
                     r["CLR_FN"], r["Frag"], r["MT"], r["ML"], r["CLR_Pr"], r["CLR_Re"]])
    hi = {1: True, 2: True, 3: True, 4: True, 5: True, 6: False, 7: False, 8: False, 9: False, 10: True, 11: False,
          12: True, 13: True}
    rows = bold_best(rows, list(hi), hi)
    return md(["Method", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag", "MT", "ML", "Prec.",
               "Rec."], rows)


def partial_label_table(summary: list[dict], group: str, param_set: str) -> str:
    """Table for a partially labelled dataset: recall- and association-based metrics first.

    Metrics that count false positives (HOTA, DetA, MOTA, IDF1) are listed last and marked,
    because unlabelled real people are counted as false positives.
    """
    rows = []
    for a in ALGORITHMS:
        r = next((x for x in summary if x["group"] == group and x["tracker"] == a and x["param_set"] == param_set), None)
        if r is None:
            return f"_No results for {group}._"
        rows.append([TRACKER_LABELS[a], r["OWTA"], r["DetRe"], r["AssA"], r["IDSW"], r["Frag"], r["MT"], r["ML"],
                     r["CLR_FN"], r["HOTA"], r["MOTA"], r["IDF1"], r["CLR_FP"]])
    hi = {1: True, 2: True, 3: True, 4: False, 5: False, 6: True, 7: False, 8: False}
    rows = bold_best(rows, list(hi), hi)
    return md(["Method", "OWTA", "DetRe", "AssA", "IDSW", "Frag", "MT", "ML", "FN", "HOTA*", "MOTA*", "IDF1*", "FP*"],
              rows)


def build_tables(manifest) -> dict[str, str]:
    m = manifest.results_dir / "metrics"
    t: dict[str, str] = {}

    # sequences
    rows = []
    for s in manifest.all_sequences():
        d = s.describe()
        rows.append([d["dataset"], f"`{d['sequence']}`", d["role"], d["frames_evaluated"],
                     f"{d['source_first_frame']}-{d['source_last_frame']}", f"{d['width']}x{d['height']}",
                     d["player_gt_tracks"] or "n/a", d["player_gt_boxes"] or "n/a",
                     d["ball_visible_frames"] if d["has_ball_boxes"] else "n/a"])
    t["sequences"] = md(["Dataset", "Sequence", "Role", "Frames", "Source frames", "Resolution", "Player tracks (GT)",
                         "Player boxes (GT)", "Visible-ball frames (GT)"], rows)

    if (m / "player_tracking_summary.csv").is_file():
        summary = read_csv(m / "player_tracking_summary.csv")
        per_seq = read_csv(m / "player_tracking_per_sequence.csv")
        sel = read_json(m / "player_selected_params.json")
        for key, group in (("tracking_all", "All test sequences"), ("tracking_sportsmot", "SportsMOT (test)"),
                           ("tracking_uvy", "UVY-Track (test)"), ("tracking_dev", "SportsMOT (dev, used for tuning)")):
            t[key] = tracking_table(summary, group, "dev-tuned")
            t[key + "_default"] = tracking_table(summary, group, "default")
        t["tracking_uvy_partial"] = partial_label_table(summary, "UVY-Track (test)", "dev-tuned")
        t["tracking_uvy_partial_default"] = partial_label_table(summary, "UVY-Track (test)", "default")

        rows = []
        seqs = list(dict.fromkeys((r["dataset"], r["sequence"], r["role"]) for r in per_seq))
        for ds, name, role in seqs:
            for a in ALGORITHMS:
                r = next(x for x in per_seq if (x["dataset"], x["sequence"], x["tracker"], x["param_set"]) ==
                         (ds, name, a, "dev-tuned"))
                rows.append([ds, f"`{name}`", role, SHORT_LABELS[a], r["HOTA"], r["OWTA"], r["DetA"], r["AssA"],
                             r["MOTA"], r["IDF1"], r["IDSW"], r["Frag"], r["CLR_FP"], r["CLR_FN"], r["gt_tracks"],
                             r["pred_tracks"]])
        t["tracking_per_sequence"] = md(["Dataset", "Sequence", "Role", "Tracker", "HOTA", "OWTA", "DetA", "AssA",
                                         "MOTA", "IDF1", "IDSW", "Frag", "FP", "FN", "GT tracks", "Output tracks"],
                                        rows)

        names = ["lost_track_buffer", "minimum_consecutive_frames", "minimum_iou_threshold",
                 "track_activation_threshold", "high_conf_det_threshold"]
        rows = [["Project default (all trackers)"] + [sel["default"][k] for k in names]]
        for a in ALGORITHMS:
            rows.append([f"{SHORT_LABELS[a]}, dev-tuned"] +
                        [sel["selected"][a][k] if k in accepted_params(a) else "not used" for k in names])
        t["selected_params"] = md(["Parameter set"] + [f"`{n}`" for n in names], rows)

        rows = []
        for a in ALGORITHMS:
            r = next(x for x in summary if x["group"] == "All test sequences" and x["tracker"] == a
                     and x["param_set"] == "dev-tuned")
            rows.append([TRACKER_LABELS[a], r["tracker_ms_per_frame"], r["tracker_only_fps"]])
        t["tracker_latency"] = md(["Method", "Tracking stage (ms / frame)", "Tracking stage alone (FPS)"], rows)

    if (m / "player_detection.csv").is_file():
        det = read_csv(m / "player_detection.csv")
        rows = []
        for ds, role in dict.fromkeys((r["dataset"], r["role"]) for r in det):
            rs = [r for r in det if (r["dataset"], r["role"]) == (ds, role)]
            tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
            p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            ap = sum(r["ap50"] for r in rs) / len(rs)
            rows.append([ds, role, len(rs), tp, fp, fn, 100 * p, 100 * rc, 100 * 2 * p * rc / max(p + rc, 1e-9), ap])
        t["player_detection"] = md(["Dataset", "Role", "Sequences", "TP", "FP", "FN", "Precision", "Recall", "F1",
                                    "AP@0.5 (mean over sequences)"], rows)

    if (m / "ball_summary.csv").is_file():
        bs = read_csv(m / "ball_summary.csv")
        bp = read_csv(m / "ball_per_sequence.csv")
        rows = []
        for r in bs:
            rows.append([r["group"], r["method_label"], r["frames_ball_visible"], r["precision_dist"], r["recall_dist"],
                         r["f1_dist"], r["precision_iou"], r["recall_iou"], r["f1_iou"], r.get("ap50_candidates"),
                         r["coverage"], r.get("center_error_correct_px")])
        t["ball_summary"] = md(["Data", "Method", "Visible-ball frames", "P (dist)", "R (dist)", "F1 (dist)",
                                "P (IoU)", "R (IoU)", "F1 (IoU)", "AP@0.5", "Coverage",
                                "Centre error of correct outputs (px)"], rows)
        rows = []
        for r in bs:
            if r["method"] != "kalman_motion":
                continue
            rows.append([r["group"], r["frames_detected"], r.get("precision_detected_dist"), r["frames_predicted"],
                         r.get("precision_predicted_dist"), r["frames_lost"]])
        t["ball_predicted"] = md(["Data", "Detected frames", "Precision of detected positions (dist)",
                                  "Predicted frames", "Precision of predicted positions (dist)",
                                  "Frames with no output"], rows)
        rows = []
        for r in bs:
            rows.append([r["group"], r["method_label"], r["frames_detected"], r["frames_predicted"], r["frames_lost"],
                         r["track_segments"], r["track_breaks"], r["longest_segment_frames"],
                         r["mean_segment_frames"], r.get("visible_frames_gained_vs_detection_only"),
                         r.get("visible_frames_lost_vs_detection_only"), r.get("id_switches")])
        t["ball_continuity"] = md(["Data", "Method", "Frames detected", "Frames predicted", "Frames lost",
                                   "Correct segments", "Breaks", "Longest segment (frames)",
                                   "Mean segment (frames)", "Visible frames gained vs det.-only",
                                   "Visible frames lost vs det.-only", "Ball ID switches"], rows)
        rows = []
        for r in bp:
            if r["method"] == "kalman_detected":
                continue
            rows.append([r["dataset"], f"`{r['sequence']}`", r["method_label"], r["frames_labelled"],
                         r["frames_ball_visible"], r["precision_dist"], r["recall_dist"], r["f1_dist"],
                         r["frames_predicted"], r.get("precision_predicted_dist"), r["center_error_mean_px"]])
        t["ball_per_sequence"] = md(["Dataset", "Sequence", "Method", "Frames", "Visible-ball frames", "P (dist)",
                                     "R (dist)", "F1 (dist)", "Predicted frames", "Precision of predicted frames",
                                     "Centre error (px)"], rows)

    if (m / "runtime_benchmark.json").is_file():
        b = read_json(m / "runtime_benchmark.json")
        total = sum(v["median_ms"] for k, v in b["stages"].items()
                    if not (k.startswith("player filter") and "ByteTrack" not in k))
        rows = [[k, v["median_ms"], v["mean_ms"], v["p95_ms"],
                 "n/a (baseline)" if (k.startswith("player filter") and "ByteTrack" not in k)
                 else f"{100 * v['median_ms'] / total:.1f}%"] for k, v in b["stages"].items()]
        t["runtime_stages"] = md(["Stage", "Median (ms / frame)", "Mean (ms / frame)", "95th percentile (ms)",
                                  "Share of proposed pipeline"], rows)
        rows = [[TRACKER_LABELS[a], b["full_pipeline_ms_per_frame"][a], b["full_pipeline_fps"][a]] for a in ALGORITHMS]
        t["runtime_pipeline"] = md(["Full pipeline with", "Total time (ms / frame)", "Throughput (FPS)"], rows)
        s = b["system"]
        rows = [["CPU", s.get("cpu")], ["Cores (physical / logical)", f"{s.get('cpu_physical_cores')} / {s.get('cpu_logical_cores')}"],
                ["RAM", f"{s.get('ram_gb')} GB"], ["GPU", s.get("gpu") or "none (CUDA not available)"],
                ["Inference device", s.get("device")], ["Power source during the benchmark", s.get("power_source")],
                ["Operating system", f"{s.get('platform')} (Windows 11)" if "10.0.2" in str(s.get("platform"))
                 else s.get("platform")],
                ["Python", s.get("python")], ["PyTorch", f"{s.get('torch')} ({s.get('torch_threads')} threads)"],
                ["Ultralytics", s.get("ultralytics")], ["Roboflow trackers", "2.6.1"],
                ["supervision", s.get("supervision")], ["OpenCV", s.get("cv2")],
                ["Benchmark sequence", f"{b['sequence']}, {b['frames']} frames, {b['resolution']}"]]
        t["system"] = md(["Item", "Value"], rows)

    if (m / "model_comparison.csv").is_file():
        rows = [[r["model"], r["target"], r["dataset"], r["frames"], r["precision"], r["recall"], r["f1"], r["ap50"],
                 r["detector_ms_per_frame"]] for r in read_csv(m / "model_comparison.csv")]
        t["model_comparison"] = md(["Model", "Target", "Dataset", "Frames", "Precision", "Recall", "F1", "AP@0.5",
                                    "Detector (ms / frame)"], rows)
    return t


# --------------------------------------------------------------------------- #
# results/tables.md: exactly the tables and figures of the IEEE paper, in the
# paper's order, with the paper's columns and rounding. The rows come from the
# same builder that fills the LaTeX paper (scripts/make_latex_report.py, used
# read-only), so tables.md and the PDF always show the same numbers.
# --------------------------------------------------------------------------- #
LATEX_TEMPLATE = Path("docs/latex/FINAL_REPORT.template.tex")

# (kind, title, content) in the order they appear in the paper.
# kind "tab": content = (key in the LaTeX builder, header); "fig": content = name of the function below
# that turns the figure's data into a table. Figs. 1 and 9 (the architecture diagram and a frame of
# the output video) contain no data and are not listed.
PAPER_ITEMS = [
    ("static", "Table I. Datasets and how they are used", "Datasets and How They Are Used"),
    ("tab", "Table II. Evaluated sequences", ("TAB_SEQUENCES",
        ["Dataset", "Sequence", "Role", "Frames", "Tracks", "Ball fr."])),
    ("tab", "Table III. Tracker parameters (n/a: parameter not used by the tracker)", ("TAB_PARAMS",
        ["Parameter set", "Buffer (frames)", "Confirm (frames)", "Min. IoU", "New-track threshold",
         "High-conf. threshold"])),
    ("tab", "Table IV. Player detection with YOLO11n (confidence >= 0.5, IoU >= 0.5)", ("TAB_DETECTION",
        ["Dataset", "Role", "Seq.", "Prec.", "Rec.", "F1", "AP@0.5"])),
    ("fig", "Fig. 2. Tracking accuracy of the three trackers (tuned parameters)", "fig_tracking_accuracy"),
    ("fig", "Fig. 3. Identity switches and track fragmentations", "fig_idsw_frag"),
    ("tab", "Table V. Player tracking on the SportsMOT test sequences (identical YOLO11n detections)",
        ("TAB_SPORTSMOT", ["Method", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Frag", "MT",
                           "Prec.", "Rec."])),
    ("fig", "Fig. 4. HOTA on the dev sequence for the 48 grid points of each tracker", "fig_tuning"),
    ("tab", "Table VI. Player tracking on the UVY-Track test sequences (tuned parameters)", ("TAB_UVY",
        ["Tracker", "OWTA", "DetRe", "AssA", "IDSW", "Frag", "MT", "FN"])),
    ("tab", "Table VII. Ball results: detection only vs the Kalman ball tracker with and without motion flights",
        ("TAB_BALL", ["Data", "Method", "Prec. (dist)", "Rec. (dist)", "F1 (dist)", "Prec. (IoU)", "Rec. (IoU)",
                      "F1 (IoU)", "AP@0.5", "Cover.", "Frames det.", "Frames pred.", "Frames none",
                      "Correct segments", "Mean seg. len."])),
    ("fig", "Fig. 5. Precision, recall and F1 for players and ball", "fig_detection_prf"),
    ("tab", "Table VIII. Proposed ball tracker: precision of detected and of predicted positions", ("TAB_BALLPRED",
        ["Data", "Detected frames", "Detected prec. (%)", "Predicted frames", "Predicted prec. (%)"])),
    ("fig", "Fig. 6. Ball trajectory of the proposed tracker on the free kick of soccer_V04", "fig_ball_trajectory"),
    ("fig", "Fig. 7. Ball track status per sequence with the proposed tracker", "fig_ball_status"),
    ("tab", "Table IX. YOLOv8n versus YOLO11n on the same frames", ("TAB_MODELS",
        ["Model", "Target", "Frames", "Prec.", "Rec.", "F1", "AP@0.5", "ms/fr."])),
    ("tab", "Table X. Median processing time per frame", ("TAB_RUNTIME", ["Stage", "Time (ms)", "Share (%)"])),
    ("fig", "Fig. 8. Runtime", "fig_runtime"),
    ("fig", "Fig. 10. Player trajectories from ByteTrack (SportsMOT test sequence)", "fig_player_trajectories"),
    ("fig", "Fig. 11. Player-position heatmap", "fig_player_heatmap"),
]


def _latex_cell_to_md(cell: str) -> str:
    """Plain text of one LaTeX table cell, as the PDF shows it."""
    s = cell.strip()
    s = re.sub(r"~?\\cite\{[^}]*\}", "", s)
    s = s.replace(r"$^{\dagger}$", "†").replace("{,}", ",").replace(r"\_", "_").replace(r"\%", "%")
    s = s.replace("``", '"').replace("''", '"')
    for cmd in ("textbf", "texttt", "textit", "emph"):
        s = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", lambda m: f"**{m.group(1)}**" if cmd == "textbf" else m.group(1), s)
    s = s.replace("--", "–")
    return s.replace("|", "/").strip()


def _latex_rows_to_md(latex: str, header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for raw in latex.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith(r"\midrule") or raw.startswith(r"\bottomrule"):
            continue
        group = re.match(r"\\multicolumn\{\d+\}\{\w+\}\{(.*)\}\s*\\\\$", raw)
        if group:  # a sub-heading row spanning the table, e.g. "Parameters tuned per tracker ..."
            lines.append("| *" + _latex_cell_to_md(group.group(1)) + "* |" + " |" * (len(header) - 1))
            continue
        cells = [_latex_cell_to_md(c) for c in re.sub(r"\\\\\s*$", "", raw).split(" & ")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _static_table_from_template(caption: str) -> str:
    """A descriptive table typed in the paper itself (Table I), copied from the LaTeX template."""
    tex = LATEX_TEMPLATE.read_text(encoding="utf-8")
    block = tex[tex.index(r"\caption{" + caption + "}"):]
    body = block[block.index(r"\toprule") + len(r"\toprule"):block.index(r"\bottomrule")]
    rows = [r.strip() for r in body.split("\n") if r.strip() and not r.strip().startswith(r"\midrule")]
    header, rest = rows[0], rows[1:]
    head = [_latex_cell_to_md(c) for c in re.sub(r"\\\\\s*$", "", header).split("&")]
    return _latex_rows_to_md("\n".join(rest), head)


# --------------------------------------------------------------------------- #
# Figures as tables. Each function reads the same result files and does the
# same computation as the matching plot in soccer_tracker/experiments/plots.py,
# with the rounding of the plot's labels; nothing is typed in by hand.
# --------------------------------------------------------------------------- #
def _f(v, nd: int = 1) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def fig_tracking_accuracy(manifest) -> str:
    from soccer_tracker.experiments.plots import _row, _summary

    rows, parts = _summary(manifest), []
    for group, metrics in (("SportsMOT (test)", ["HOTA", "MOTA", "IDF1"]), ("UVY-Track (test)", ["OWTA", "AssA", "DetRe"])):
        if not any(r["group"] == group for r in rows):
            continue
        b = _row(rows, group, "bytetrack")
        body = [[SHORT_LABELS[a]] + [_f(_row(rows, group, a)[m]) for m in metrics] for a in ALGORITHMS]
        parts.append(f"**{group}: {b['sequences']} sequences, {b['frames']} frames**\n\n" + md(["Tracker"] + metrics, body))
    return "\n\n".join(parts)


def fig_idsw_frag(manifest) -> str:
    from soccer_tracker.experiments.plots import _row, _summary

    rows = _summary(manifest)
    groups = [g for g in ("SportsMOT (test)", "UVY-Track (test)") if any(r["group"] == g for r in rows)]
    head = ["Tracker"] + [f"IDSW, {g.replace(' (test)', '')}" for g in groups] + \
           [f"Frag, {g.replace(' (test)', '')}" for g in groups]
    body = [[SHORT_LABELS[a]] + [_f(_row(rows, g, a)["IDSW"], 0) for g in groups]
            + [_f(_row(rows, g, a)["Frag"], 0) for g in groups] for a in ALGORITHMS]
    return md(head, body) + "\n\nLower is better."


def fig_tuning(manifest) -> str:
    m = manifest.results_dir / "metrics"
    rows = read_csv(m / "player_tuning_grid.csv")
    sel = read_json(m / "player_selected_params.json")
    body = []
    for a in ALGORITHMS:
        vals = [r["HOTA"] for r in rows if r["tracker"] == a]
        dflt = next((r["HOTA"] for r in rows if r["tracker"] == a
                     and all(r[k] == v for k, v in sel["default"].items())), None)
        srt = sorted(vals)
        body.append([SHORT_LABELS[a], len(vals), _f(srt[0]), _f(srt[len(srt) // 2]), _f(srt[-1]), _f(dflt)])
    return (f"Dev sequence: {', '.join(sel['dev_sequences'])}. Selected = best HOTA (star in the figure); "
            "default = shared project parameters (circle).\n\n"
            + md(["Tracker", "Grid points", "Lowest HOTA", "Median HOTA", "Selected (best) HOTA", "Default-parameter HOTA"],
                 body))


def fig_detection_prf(manifest) -> str:
    m = manifest.results_dir / "metrics"
    det = [r for r in read_csv(m / "player_detection.csv") if r["role"] == "test"]
    ball = read_csv(m / "ball_summary.csv")
    body = []
    for ds in dict.fromkeys(r["dataset"] for r in det):
        rs = [r for r in det if r["dataset"] == ds]
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        if ds == "UVY-Track":  # partially labelled: only recall can be measured
            body.append([ds + " (recall only)", "–", _f(100 * rc), "–"])
        else:
            body.append([ds, _f(100 * p), _f(100 * rc), _f(100 * 2 * p * rc / max(p + rc, 1e-9))])
    parts = ["**Players: YOLO11n detector (confidence >= 0.5, IoU >= 0.5)**\n\n"
             + md(["Dataset", "Precision", "Recall", "F1"], body)]
    for group in (g for g in ("UVY-Track", "InFactory") if any(r["group"] == g for r in ball)):
        rows, body = {r["method"]: r for r in ball if r["group"] == group}, []
        for method, label in (("detection_only", "Detection only"), ("kalman", "Kalman tracker (YOLO only)"),
                              ("kalman_motion", "Kalman + motion flights (proposed)")):
            r = rows[method]
            body.append([label, _f(r["precision_dist"]), _f(r["recall_dist"]), _f(r["f1_dist"])])
        parts.append(f"**Ball on {group}: {rows['kalman_motion']['frames_ball_visible']} visible-ball frames "
                     "(centre within one ball diameter)**\n\n" + md(["Method", "Precision", "Recall", "F1"], body))
    return "\n\n".join(parts)


def fig_ball_trajectory(manifest, step: int = 10) -> str:
    from soccer_tracker.experiments.plots import _ball_trace_sequence

    seq, rows = _ball_trace_sequence(manifest)
    count = {s: sum(1 for r in rows if r["mot_status"] == s) for s in ("detected", "predicted", "lost")}
    head = (f"{seq.info.dataset} {seq.info.name}, {len(rows)} frames: {count['detected']} detected (by the detector "
            f"or a motion flight), {count['predicted']} predicted by the Kalman filter, {count['lost']} with no output; "
            f"ground truth visible in {sum(1 for r in rows if r['ball_visible'])} frames. Every {step}th frame:")
    body = [[r["frame"], _f(r["gt_cx"], 0), _f(r["gt_cy"], 0), r["mot_status"], _f(r["mot_cx"], 0), _f(r["mot_cy"], 0)]
            for r in rows if r["frame"] % step == 0]
    return head + "\n\n" + md(["Frame", "Ground truth x", "Ground truth y", "Tracker status", "Tracker x",
                                 "Tracker y"], body)


def fig_ball_status(manifest) -> str:
    rows = [r for r in read_csv(manifest.results_dir / "metrics" / "ball_per_sequence.csv")
            if r["method"] == "kalman_motion" and r["frames_labelled"] >= 30]
    body = [[f"{r['sequence']} ({r['dataset']})", r["frames_labelled"]]
            + [f"{100.0 * r[k] / r['frames_labelled']:.0f}%" for k in
               ("frames_detected", "frames_predicted", "frames_lost", "frames_ball_visible")] for r in rows]
    return md(["Sequence", "Labelled frames", "Detected", "Predicted (Kalman)", "Lost (no output)",
               "Frames with a visible ball"], body)


def fig_runtime(manifest) -> str:
    from soccer_tracker.experiments.plots import _row, _summary

    rows = _summary(manifest)
    bench = read_json(manifest.results_dir / "metrics" / "runtime_benchmark.json")
    body = [[SHORT_LABELS[a], _f(_row(rows, "All test sequences", a)["tracker_ms_per_frame"], 2),
             _f(bench["full_pipeline_fps"][a], 2)] for a in ALGORITHMS]
    stages = sorted(bench["stages"].items(), key=lambda kv: -kv[1]["median_ms"])
    stage_rows = [[k, _f(v["median_ms"], 2)] for k, v in stages
                  if not (k.startswith("player filter") and "ByteTrack" not in k)]
    return ("**Left and centre panels**\n\n"
            + md(["Tracker", "Tracking stage only (ms / frame, mean over all test sequences)",
                  "Full pipeline (frames per second)"], body)
            + f"\n\n**Right panel: where the time goes ({bench['frames']} frames of {bench['sequence']})**\n\n"
            + md(["Stage", "Median time (ms / frame)"], stage_rows))


def fig_player_trajectories(manifest, window: int = 75) -> str:
    from soccer_tracker.experiments.plots import _overlay_sequence, _proposed_tracks

    seq = _overlay_sequence(manifest)
    tr = _proposed_tracks(manifest, seq)
    end = min(seq.info.num_frames, 150)
    start = max(1, end - window + 1)
    body = []
    for tid in sorted(set(tr[:, 1].astype(int))):
        t = tr[(tr[:, 1] == tid) & (tr[:, 0] >= start) & (tr[:, 0] <= end)]
        if len(t) < 5:  # same rule as the figure
            continue
        fx, fy = t[:, 2] + t[:, 4] / 2, t[:, 3] + t[:, 5]
        dist = float(((fx[1:] - fx[:-1]) ** 2 + (fy[1:] - fy[:-1]) ** 2).__pow__(0.5).sum())
        body.append([f"#{tid}", int(t[0, 0]), int(t[-1, 0]), len(t), f"({fx[0]:.0f}, {fy[0]:.0f})",
                     f"({fx[-1]:.0f}, {fy[-1]:.0f})", f"{dist:.0f}"])
    return (f"{seq.info.dataset} {seq.info.name}, frames {start}-{end}: {len(body)} tracks (foot point = bottom "
            "centre of the box, image pixels).\n\n"
            + md(["Track", "First frame", "Last frame", "Points", "Start (x, y)", "End (x, y)", "Path length (px)"],
                 body))


def fig_player_heatmap(manifest, cols: int = 4, rows_: int = 3) -> str:
    from soccer_tracker.experiments.plots import _overlay_sequence, _proposed_tracks

    seq = _overlay_sequence(manifest)
    tr = _proposed_tracks(manifest, seq)
    w, h = seq.info.width, seq.info.height
    fx, fy = tr[:, 2] + tr[:, 4] / 2, tr[:, 3] + tr[:, 5]
    total = len(tr)
    cw, ch = w / cols, h / rows_
    body = []
    for j in range(rows_):
        line = [f"y {j * ch:.0f}-{(j + 1) * ch:.0f}"]
        for i in range(cols):
            n = int(((fx >= i * cw) & (fx < (i + 1) * cw) & (fy >= j * ch) & (fy < (j + 1) * ch)).sum())
            line.append(f"{100.0 * n / max(total, 1):.1f}%")
        body.append(line)
    return (f"{seq.info.dataset} {seq.info.name}, {seq.info.num_frames} frames, {total} tracked foot points "
            f"(ByteTrack). Share of foot points in each region of the {w}x{h} image (the figure shows the same points "
            "at 8x8-pixel resolution):\n\n"
            + md(["Image region"] + [f"x {i * cw:.0f}-{(i + 1) * cw:.0f}" for i in range(cols)], body))


def write_paper_tables(manifest, out: Path) -> tuple[int, int]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_latex_report", Path(__file__).with_name("make_latex_report.py"))
    latex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(latex)
    paper = latex.build_tables(manifest)

    parts = ["# Tables and figures of the IEEE paper, as tables (generated by scripts/make_report_tables.py)",
             "Same order as docs/latex/FINAL_REPORT.tex. Tables have the paper's columns and rounding; each figure "
             "is given as a table of the numbers it plots. Every number is computed from the results of the last run "
             "(results/metrics/, results/tracks/). Figs. 1 and 9 (the architecture diagram and a frame of the output "
             "video) contain no data. Lower is better for IDSW, FP, FN and Frag; bold marks the best value."]
    n_tab = n_fig = 0
    for kind, title, content in PAPER_ITEMS:
        if kind == "fig":
            parts.append(f"## {title}\n\n{globals()[content](manifest)}")
            n_fig += 1
        elif kind == "static":
            parts.append(f"## {title}\n\n{_static_table_from_template(content)}")
            n_tab += 1
        else:
            key, header = content
            text = _latex_rows_to_md(paper[key], header)
            if key == "TAB_DETECTION":
                text += "\n\n† Not meaningful: only part of the people are labelled."
            parts.append(f"## {title}\n\n{text}")
            n_tab += 1
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return n_tab, n_fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    ap.add_argument("--target", action="append", default=None,
                    help="Markdown file(s) whose table blocks are refreshed (default: the report and the README)")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    tables = build_tables(manifest)

    out = manifest.results_dir / "tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_tables, n_figures = write_paper_tables(manifest, out)
    print(f"{n_tables} tables and {n_figures} figures of the IEEE paper written to {out}")

    for target in args.target or ["docs/FINAL_REPORT.md", "README.md"]:
        path = Path(target)
        if not path.is_file():
            continue
        missing: list[str] = []

        def sub(match: re.Match) -> str:
            name = match.group(1)
            if name not in tables:
                missing.append(name)
                return match.group(0)
            return f"<!-- TABLE:{name} -->\n{tables[name]}\n<!-- /TABLE:{name} -->"

        text = path.read_text(encoding="utf-8")
        pattern = r"<!-- TABLE:(\w+) -->.*?<!-- /TABLE:\1 -->"
        path.write_text(re.sub(pattern, sub, text, flags=re.S), encoding="utf-8")
        print(f"Tables refreshed in {path}" + (f"; no data for: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
