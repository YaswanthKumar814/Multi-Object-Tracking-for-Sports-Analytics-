r"""Build the final report as an IEEE-format LaTeX paper.

    python scripts/make_latex_report.py

Writes
    docs/latex/FINAL_REPORT.tex          IEEEtran conference paper, one self-contained file
    docs/latex/figures/*.png             the figures it includes
    docs/FINAL_REPORT_IEEE_overleaf.zip  the two above, ready for Overleaf ("Upload Project")

Every table is generated from results/metrics/, so no number in a table is typed
by hand. Numbers quoted in the running text are checked against the metrics files
and a warning is printed if the results have changed since the text was written.
The script also lints the generated file (braces, environments, table columns,
unescaped special characters, missing figures), because a LaTeX compiler may not
be installed locally.

Compile with:  pdflatex FINAL_REPORT.tex   (run twice), or upload the zip to Overleaf.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

from soccer_tracker.experiments import load_manifest
from soccer_tracker.experiments.common import read_csv, read_json
from soccer_tracker.tracking.factory import ALGORITHMS, accepted_params

NAMES = {"sort": "YOLO11n + SORT", "ocsort": "YOLO11n + OC-SORT", "bytetrack": "YOLO11n + ByteTrack (proposed)"}
SHORT = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack (proposed)"}

FIGURES = {
    "fig_architecture.png": "plots/00_architecture.png",
    "fig_tracking_accuracy.png": "plots/01_hota_mota_idf1.png",
    "fig_detection_prf.png": "plots/03_detection_prf.png",
    "fig_runtime.png": "plots/04_runtime.png",
    "fig_idsw_frag.png": "plots/05_idsw_frag.png",
    "fig_player_trajectories.png": "plots/06_player_trajectories.png",
    "fig_player_heatmap.png": "plots/07_player_heatmap.png",
    "fig_ball_trajectory.png": "plots/08_ball_trajectory.png",
    "fig_ball_status.png": "plots/09_ball_status.png",
    "fig_tuning.png": "plots/12_tuning_sensitivity.png",
    "fig_demo_screenshot.png": "demo/uvy_soccer_V04_screenshot.png",
}


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    out = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        out = out.replace(a, b)
    return out


def tt(text: str) -> str:
    return r"\texttt{" + esc(text) + "}"


def num(v, nd: int = 1) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    return f"{v:.{nd}f}"


def best(values: list, higher: bool = True, nd: int = 1) -> list[str]:
    """Format a column, bolding the best value (ties all bold)."""
    nums = [v for v in values if v is not None]
    target = (max(nums) if higher else min(nums)) if nums else None
    return [(r"\textbf{" + num(v, nd) + "}") if (v is not None and v == target) else num(v, nd) for v in values]


def row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def tracking_block(summary: list[dict], group: str, param_set: str, cols: list[tuple[str, bool, int]],
                   names: dict[str, str] = NAMES) -> list[str]:
    rows = [next(r for r in summary if r["group"] == group and r["tracker"] == a and r["param_set"] == param_set)
            for a in ALGORITHMS]
    columns = [best([r[key] for r in rows], higher, nd) for key, higher, nd in cols]
    return [row([names[a]] + [c[i] for c in columns]) for i, a in enumerate(ALGORITHMS)]


def build_tables(manifest) -> dict[str, str]:
    m = manifest.results_dir / "metrics"
    summary = read_csv(m / "player_tracking_summary.csv")
    sel = read_json(m / "player_selected_params.json")
    det = read_csv(m / "player_detection.csv")
    ball = read_csv(m / "ball_summary.csv")
    bench = read_json(m / "runtime_benchmark.json")
    models = read_csv(m / "model_comparison.csv") if (m / "model_comparison.csv").is_file() else []
    t: dict[str, str] = {}

    # ---- sequences ----------------------------------------------------------
    seqs = manifest.all_sequences()
    lines = []
    for s in seqs:
        d = s.describe()
        if d["dataset"] == "InFactory":
            continue
        lines.append(row([d["dataset"], tt(d["sequence"]), d["role"], num(d["frames_evaluated"]),
                          num(d["player_gt_tracks"]), num(d["ball_visible_frames"]) if d["has_ball_boxes"] else "--"]))
    inf = [s.describe() for s in seqs if s.info.dataset == "InFactory"]
    if inf:
        lines.append(row(["InFactory", f"{len(inf)} clips", "test", num(sum(d["frames_evaluated"] for d in inf)), "--",
                          num(sum(d["ball_visible_frames"] for d in inf))]))
    t["TAB_SEQUENCES"] = "\n".join(lines)
    frames = {name: sum(s.info.num_frames for s in seqs if s.info.dataset == name)
              for name in ("SportsMOT", "UVY-Track", "InFactory")}
    t["FRAMES_SM"], t["FRAMES_UVY"], t["FRAMES_IF"] = (num(frames[k]) for k in ("SportsMOT", "UVY-Track", "InFactory"))
    t["FRAMES_ALL"] = num(sum(frames.values()))

    # ---- selected parameters ------------------------------------------------
    keys = ["lost_track_buffer", "minimum_consecutive_frames", "minimum_iou_threshold", "track_activation_threshold",
            "high_conf_det_threshold"]

    def pval(v):
        return str(v) if isinstance(v, int) else f"{v:.1f}"

    lines = [row(["Shared default"] + [pval(sel["default"][k]) for k in keys]), r"\midrule"]
    for a in ALGORITHMS:
        lines.append(row([SHORT[a] + ", tuned"] +
                         [pval(sel["selected"][a][k]) if k in accepted_params(a) else "n/a" for k in keys]))
    t["TAB_PARAMS"] = "\n".join(lines)

    # ---- player detection ---------------------------------------------------
    lines = []
    for ds, role in dict.fromkeys((r["dataset"], r["role"]) for r in det):
        rs = [r for r in det if (r["dataset"], r["role"]) == (ds, role)]
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        p, rc = 100 * tp / max(tp + fp, 1), 100 * tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        ap = sum(r["ap50"] for r in rs) / len(rs)
        dag = r"$^{\dagger}$" if ds == "UVY-Track" else ""
        lines.append(row([ds, role, str(len(rs)), num(p) + dag, num(rc), num(f1) + dag, num(ap) + dag]))
    t["TAB_DETECTION"] = "\n".join(lines)

    # ---- SportsMOT tracking (tuned + default) -------------------------------
    cols = [("HOTA", True, 1), ("DetA", True, 1), ("AssA", True, 1), ("MOTA", True, 1), ("IDF1", True, 1),
            ("IDSW", False, 0), ("CLR_FP", False, 0), ("CLR_FN", False, 0), ("Frag", False, 0), ("MT", True, 0),
            ("CLR_Pr", True, 1), ("CLR_Re", True, 1)]
    n = len(cols) + 1
    lines = [r"\multicolumn{%d}{l}{\textit{Parameters tuned per tracker on the dev sequence}} \\" % n]
    lines += tracking_block(summary, "SportsMOT (test)", "dev-tuned", cols)
    lines += [r"\midrule", r"\multicolumn{%d}{l}{\textit{One shared, untuned parameter set (project defaults)}} \\" % n]
    lines += tracking_block(summary, "SportsMOT (test)", "default", cols)
    t["TAB_SPORTSMOT"] = "\n".join(lines)

    # ---- UVY-Track tracking -------------------------------------------------
    cols = [("OWTA", True, 1), ("DetRe", True, 1), ("AssA", True, 1), ("IDSW", False, 0), ("Frag", False, 0),
            ("MT", True, 0), ("CLR_FN", False, 0)]
    t["TAB_UVY"] = "\n".join(tracking_block(summary, "UVY-Track (test)", "dev-tuned", cols, SHORT))

    # ---- ball ---------------------------------------------------------------
    labels = {"detection_only": "Detection only", "kalman_detected": "Kalman, detected only",
              "kalman": "Kalman (proposed)"}
    lines = []
    groups = [g for g in ("UVY-Track", "InFactory", "All ball sequences") if any(r["group"] == g for r in ball)]
    for gi, g in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        for method in ("detection_only", "kalman_detected", "kalman"):
            r = next(x for x in ball if x["group"] == g and x["method"] == method)
            lines.append(row([g.replace("All ball sequences", "All (13 seq.)"), labels[method],
                              num(r["precision_dist"]), num(r["recall_dist"]), num(r["f1_dist"]),
                              num(r["precision_iou"]), num(r["recall_iou"]), num(r["f1_iou"]),
                              num(r.get("ap50_candidates")), num(r["coverage"]),
                              num(r["frames_detected"]), num(r["frames_predicted"]), num(r["frames_lost"]),
                              num(r["track_segments"]), num(r["mean_segment_frames"])]))
    t["TAB_BALL"] = "\n".join(lines)

    lines = []
    for g in groups:
        r = next(x for x in ball if x["group"] == g and x["method"] == "kalman")
        lines.append(row([g.replace("All ball sequences", "All"), num(r["frames_detected"]),
                          num(r.get("precision_detected_dist")), num(r["frames_predicted"]),
                          num(r.get("precision_predicted_dist"))]))
    t["TAB_BALLPRED"] = "\n".join(lines)

    # ---- runtime ------------------------------------------------------------
    stages = bench["stages"]
    proposed = {k: v for k, v in stages.items() if not (k.startswith("player filter") and "ByteTrack" not in k)}
    total = sum(v["median_ms"] for v in proposed.values())
    lines = []
    for k, v in sorted(proposed.items(), key=lambda kv: -kv[1]["median_ms"]):
        name = k.replace("(JPEG decode)", "").replace("(mp4v)", "").replace(" (proposed)", "").strip()
        lines.append(row([esc(name[0].upper() + name[1:]), num(v["median_ms"], 2), num(100 * v["median_ms"] / total)]))
    lines.append(r"\midrule")
    lines.append(row([r"\textbf{Total (proposed pipeline)}", r"\textbf{" + num(total, 1) + "}", "100.0"]))
    lines.append(r"\midrule")
    for a in ("sort", "ocsort"):
        k = next(k for k in stages if k.startswith("player filter") and SHORT[a] in k)
        lines.append(row([esc("Player filter + " + SHORT[a] + " (baseline)"), num(stages[k]["median_ms"], 2), "--"]))
    t["TAB_RUNTIME"] = "\n".join(lines)
    t["FPS_BYTETRACK"] = num(bench["full_pipeline_fps"]["bytetrack"], 2)
    t["FPS_SORT"] = num(bench["full_pipeline_fps"]["sort"], 2)
    t["FPS_OCSORT"] = num(bench["full_pipeline_fps"]["ocsort"], 2)

    # ---- model comparison ---------------------------------------------------
    lines = []
    for r in models:
        lines.append(row([esc(r["model"]), r["target"], num(r["frames"]), num(r["precision"]), num(r["recall"]),
                          num(r["f1"]), num(r["ap50"]), num(r["detector_ms_per_frame"], 0)]))
    t["TAB_MODELS"] = "\n".join(lines) if lines else row(["--"] * 8)

    # ---- hardware sentence --------------------------------------------------
    s = bench["system"]
    cpu = esc(str(s.get("cpu", "CPU")).replace("(R)", "").replace("(TM)", ""))
    t["HARDWARE"] = (
        f"{cpu} laptop CPU ({s.get('cpu_physical_cores')} physical / {s.get('cpu_logical_cores')} logical cores), "
        f"{s.get('ram_gb')}~GB RAM, no GPU, Windows~11, Python~{s.get('python')}, PyTorch~{esc(s.get('torch'))} "
        f"with {s.get('torch_threads')} threads, Ultralytics~{s.get('ultralytics')}, "
        f"\\texttt{{trackers}}~2.6.1, \\texttt{{supervision}}~{s.get('supervision')} and OpenCV~{s.get('cv2')}"
    )
    return t


def prose_checks(manifest) -> list[str]:
    """Compare numbers quoted in the running text with the metrics files."""
    m = manifest.results_dir / "metrics"
    summary = read_csv(m / "player_tracking_summary.csv")
    ball = read_csv(m / "ball_summary.csv")
    bench = read_json(m / "runtime_benchmark.json")

    def trk(group, tracker, key, ps="dev-tuned"):
        return next(r for r in summary if r["group"] == group and r["tracker"] == tracker and r["param_set"] == ps)[key]

    def bl(method, key, group="All ball sequences"):
        return next(r for r in ball if r["group"] == group and r["method"] == method)[key]

    sm, uvy = "SportsMOT (test)", "UVY-Track (test)"
    expected = [
        ("SportsMOT HOTA SORT", 65.9, round(trk(sm, "sort", "HOTA"), 1)),
        ("SportsMOT HOTA OC-SORT", 59.3, round(trk(sm, "ocsort", "HOTA"), 1)),
        ("SportsMOT HOTA ByteTrack", 62.6, round(trk(sm, "bytetrack", "HOTA"), 1)),
        ("SportsMOT MOTA SORT", 64.9, round(trk(sm, "sort", "MOTA"), 1)),
        ("SportsMOT IDF1 SORT", 70.6, round(trk(sm, "sort", "IDF1"), 1)),
        ("SportsMOT IDF1 ByteTrack", 66.8, round(trk(sm, "bytetrack", "IDF1"), 1)),
        ("SportsMOT AssA SORT", 67.0, round(trk(sm, "sort", "AssA"), 1)),
        ("SportsMOT AssA ByteTrack", 66.1, round(trk(sm, "bytetrack", "AssA"), 1)),
        ("SportsMOT IDSW SORT", 60, trk(sm, "sort", "IDSW")),
        ("SportsMOT IDSW OC-SORT", 106, trk(sm, "ocsort", "IDSW")),
        ("SportsMOT IDSW ByteTrack", 51, trk(sm, "bytetrack", "IDSW")),
        ("SportsMOT FN ByteTrack", 670, trk(sm, "bytetrack", "CLR_FN")),
        ("SportsMOT recall ByteTrack", 95.9, round(trk(sm, "bytetrack", "CLR_Re"), 1)),
        ("SportsMOT extra FP ByteTrack vs SORT", 2131, trk(sm, "bytetrack", "CLR_FP") - trk(sm, "sort", "CLR_FP")),
        ("SportsMOT default IDSW ByteTrack", 43, trk(sm, "bytetrack", "IDSW", "default")),
        ("SportsMOT default MOTA OC-SORT", 71.1, round(trk(sm, "ocsort", "MOTA", "default"), 1)),
        ("SportsMOT default HOTA spread", 0.8, round(max(trk(sm, a, "HOTA", "default") for a in ALGORITHMS)
                                                     - min(trk(sm, a, "HOTA", "default") for a in ALGORITHMS), 1)),
        ("UVY OWTA ByteTrack", 56.8, round(trk(uvy, "bytetrack", "OWTA"), 1)),
        ("UVY DetRe ByteTrack", 58.7, round(trk(uvy, "bytetrack", "DetRe"), 1)),
        ("UVY IDSW ByteTrack", 23, trk(uvy, "bytetrack", "IDSW")),
        ("UVY IDSW OC-SORT", 44, trk(uvy, "ocsort", "IDSW")),
        ("Ball recall detection only", 34.5, round(bl("detection_only", "recall_dist"), 1)),
        ("Ball recall Kalman", 43.3, round(bl("kalman", "recall_dist"), 1)),
        ("Ball F1 detection only", 45.4, round(bl("detection_only", "f1_dist"), 1)),
        ("Ball F1 Kalman", 49.3, round(bl("kalman", "f1_dist"), 1)),
        ("Ball precision detection only", 66.4, round(bl("detection_only", "precision_dist"), 1)),
        ("Ball precision Kalman", 57.4, round(bl("kalman", "precision_dist"), 1)),
        ("Ball segments detection only", 168, bl("detection_only", "track_segments")),
        ("Ball segments Kalman", 47, bl("kalman", "track_segments")),
        ("Ball precision of predicted positions", 42.6, round(bl("kalman", "precision_predicted_dist"), 1)),
        ("Ball precision of tracker-accepted detections", 68.4, round(bl("kalman", "precision_detected_dist"), 1)),
        ("Ball F1 IoU detection only", 26.7, round(bl("detection_only", "f1_iou"), 1)),
        ("Ball F1 IoU Kalman", 24.6, round(bl("kalman", "f1_iou"), 1)),
        ("Ball AP50", 16.3, round(bl("detection_only", "ap50_candidates"), 1)),
        ("Ball frames gained", 353, bl("kalman", "visible_frames_gained_vs_detection_only")),
        ("Ball frames lost", 142, bl("kalman", "visible_frames_lost_vs_detection_only")),
        ("Detector share of runtime (%)", 87, round(100 * bench["stages"]["detection"]["median_ms"] / sum(
            v["median_ms"] for k, v in bench["stages"].items()
            if not (k.startswith("player filter") and "ByteTrack" not in k)))),
        ("Detector ms per frame", 287, round(bench["stages"]["detection"]["median_ms"])),
    ]
    return [f"{label}: text says {said}, metrics now give {actual}" for label, said, actual in expected if said != actual]


# --------------------------------------------------------------------------- #
# lint
# --------------------------------------------------------------------------- #
def lint(tex: str, fig_dir: Path) -> list[str]:
    problems: list[str] = []
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in tex.splitlines())  # strip comments

    # braces
    depth = 0
    for i, ch in enumerate(body):
        if ch in "{}" and (i == 0 or body[i - 1] != "\\"):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                problems.append(f"unmatched closing brace near: {body[max(0, i - 40):i + 10]!r}")
                depth = 0
    if depth:
        problems.append(f"{depth} unclosed brace(s)")

    # environments
    stack: list[str] = []
    for mt in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        kind, env = mt.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            problems.append(f"\\end{{{env}}} does not match the open environment")
    if stack:
        problems.append(f"unclosed environment(s): {stack}")

    # math delimiters
    if len(re.findall(r"(?<!\\)\$", body)) % 2:
        problems.append("odd number of $ signs")

    # table column counts
    for mt in re.finditer(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}", body, flags=re.S):
        spec, content = mt.groups()
        spec_clean = re.sub(r"@\{[^}]*\}|p\{[^}]*\}", lambda x: "p" if x.group(0).startswith("p") else "", spec)
        ncols = len(re.findall(r"[lcrp]", spec_clean))
        for line in content.split(r"\\"):
            line = re.sub(r"\\(toprule|midrule|bottomrule|hline)|\\cmidrule(\([^)]*\))?\{[^}]*\}", "", line).strip()
            if not line:
                continue
            span = sum(int(x) - 1 for x in re.findall(r"\\multicolumn\{(\d+)\}", line))
            cells = len(re.findall(r"(?<!\\)&", line)) + 1 + span
            if cells != ncols:
                problems.append(f"table row has {cells} cells, expected {ncols}: {line[:70]!r}")

    # unescaped specials outside maths, tables and commands that take raw arguments
    text = re.sub(r"\\(url|includegraphics(\[[^\]]*\])?|label|ref|cite|bibitem|graphicspath|input)\{[^}]*\}+", "", body)
    text = re.sub(r"\\begin\{(equation|align|tabular)\*?\}.*?\\end\{\1\*?\}", "", text, flags=re.S)
    text = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", "", text, flags=re.S)
    for ch in "_#&^":
        for mt in re.finditer(r"(?<!\\)" + re.escape(ch), text):
            problems.append(f"unescaped '{ch}' near: {text[max(0, mt.start() - 40):mt.start() + 15]!r}")
    if re.search(r"[^\x00-\x7F]", tex):
        bad = sorted(set(re.findall(r"[^\x00-\x7F]", tex)))
        problems.append(f"non-ASCII characters present: {bad}")

    # figures, labels, citations
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        if not (fig_dir / name).is_file():
            problems.append(f"missing figure file: {name}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for ref in re.findall(r"\\ref\{([^}]+)\}", body):
        if ref not in labels:
            problems.append(f"\\ref to undefined label: {ref}")
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for k in group.split(",")}
    problems += [f"citation without bibitem: {k}" for k in sorted(cited - bib)]
    problems += [f"bibitem never cited: {k}" for k in sorted(bib - cited)]
    if "@@" in tex:
        problems.append("unfilled placeholder(s): " + ", ".join(sorted(set(re.findall(r"@@\w+@@", tex)))))
    return problems


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    ap.add_argument("--template", default="docs/latex/FINAL_REPORT.template.tex")
    ap.add_argument("--out-dir", default="docs/latex")
    ap.add_argument("--zip", default="docs/FINAL_REPORT_IEEE_overleaf.zip")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, src in FIGURES.items():
        src_path = manifest.results_dir / src
        if src_path.is_file():
            shutil.copyfile(src_path, fig_dir / name)
        else:
            print(f"WARNING: figure source missing: {src_path}", file=sys.stderr)

    tex = Path(args.template).read_text(encoding="utf-8")
    for key, value in build_tables(manifest).items():
        tex = tex.replace(f"@@{key}@@", value)
    out = out_dir / "FINAL_REPORT.tex"
    out.write_text(tex, encoding="utf-8", newline="\n")
    print(f"Wrote {out} ({len(tex.splitlines())} lines)")

    stale = prose_checks(manifest)
    for line in stale:
        print("WARNING (text out of date): " + line, file=sys.stderr)
    problems = lint(tex, fig_dir)
    for line in problems:
        print("LINT: " + line, file=sys.stderr)
    print(f"Checks: {len(stale)} out-of-date number(s) in the text, {len(problems)} lint problem(s)")

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out, "FINAL_REPORT.tex")
        for name in FIGURES:
            if (fig_dir / name).is_file():
                zf.write(fig_dir / name, f"figures/{name}")
    print(f"Wrote {args.zip}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
