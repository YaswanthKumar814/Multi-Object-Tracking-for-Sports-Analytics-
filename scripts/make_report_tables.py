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
            if r["method"] != "kalman":
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
    body = "\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items())
    out.write_text("# Result tables (generated by scripts/make_report_tables.py)\n\n" + body + "\n", encoding="utf-8")
    print(f"{len(tables)} tables written to {out}")

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
