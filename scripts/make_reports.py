r"""Build the project report as two LaTeX documents with identical content.

    python scripts/make_reports.py

Writes, under docs/report/:
    report_ieee.tex             IEEE conference format, two columns (IEEEtran)
    report_single_column.tex    single-column article
    figures/*.png               the figures both documents include
and one Overleaf-ready zip per document (the .tex file plus figures/).

The text lives in docs/report/report_body.template.tex. Every table and every
number quoted in the text is filled in here from results/ (``@@NAME@@``
placeholders), so the reports cannot drift from the stored results. Both files
are linted (braces, environments, table columns, special characters, figures,
references, citations) because a LaTeX compiler may not be installed locally.

Compile either file with ``pdflatex`` (run twice), or upload its zip to Overleaf.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path

from soccer_tracker.experiments import load_manifest
from soccer_tracker.experiments.common import read_csv, read_json
from soccer_tracker.tracking.factory import ALGORITHMS

ROOT = Path(__file__).resolve().parent.parent

# Reuse the table builders and the linter of the original paper generator.
_spec = importlib.util.spec_from_file_location("make_latex_report", ROOT / "scripts" / "make_latex_report.py")
mlr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlr)

FIGURES = dict(mlr.FIGURES)
FIGURES["fig_sota.png"] = "plots/14_sota_comparison.png"

TITLE = (r"Multi-Object Tracking for Sports Analytics:\\ Player and Ball Tracking with YOLO11n, ByteTrack "
         r"and a Kalman Filter")

AUTHORS = [
    ("Ravva Siddhartha", "BL.EN.U4CSE23246"),
    ("Pulapa Yaswanth Kumar", "BL.EN.U4CSE23245"),
    ("Dr. Tripty Singh", None),
]
AFFILIATION = [r"Dept. of Computer Science and Engineering", r"Amrita School of Computing",
               r"Amrita Vishwa Vidyapeetham", r"Bengaluru, India"]

PREAMBLE_COMMON = r"""\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{adjustbox}
\usepackage{url}
\graphicspath{{figures/}}
"""


def ieee_front() -> str:
    blocks = []
    for name, reg in AUTHORS:
        lines = [r"\textit{" + AFFILIATION[0] + "}", r"\textit{" + AFFILIATION[1] + "}"] + AFFILIATION[2:]
        if reg:
            lines.append(r"\texttt{" + reg + "}")
        blocks.append(r"\IEEEauthorblockN{" + name + "}\n" + r"\IEEEauthorblockA{" + "\\\\\n".join(lines) + "}")
    return (r"\documentclass[conference]{IEEEtran}" + "\n" + r"\IEEEoverridecommandlockouts" + "\n" + PREAMBLE_COMMON
            + "\n\\begin{document}\n\n\\title{" + TITLE + "}\n\n\\author{\n" + "\n\\and\n".join(blocks) + "\n}\n\n"
            "\\maketitle\n")


def single_front() -> str:
    people = []
    for name, reg in AUTHORS:
        people.append(name + (r" \\ \small\texttt{" + reg + "}" if reg else r" \\ \small Project advisor"))
    affiliation = r" \\ ".join(AFFILIATION)
    return (r"\documentclass[11pt,a4paper]{article}" + "\n" + r"\usepackage[margin=2.5cm]{geometry}" + "\n"
            + PREAMBLE_COMMON
            + r"\usepackage{titlesec}" + "\n"
            + r"\titleformat{\section}{\large\bfseries}{\thesection.}{0.6em}{}" + "\n"
            + r"\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}" + "\n"
            + r"\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}" + "\n"
            + r"\newenvironment{IEEEkeywords}{\par\smallskip\noindent\textbf{\textit{Index Terms}---}}{\par}" + "\n"
            + r"\setlength{\parskip}{0.35em}" + "\n"
            + "\n\\begin{document}\n\n\\title{" + TITLE.replace(r"\\", r"\\[0.3em]") + "}\n\n"
            + "\\author{" + r" \and ".join(people) + r" \\[0.8em] \small " + affiliation + "}\n"
            + "\\date{}\n\n\\maketitle\n")


# --------------------------------------------------------------------------- #
# numbers quoted in the text
# --------------------------------------------------------------------------- #
def numbers(manifest) -> dict[str, str]:
    m = manifest.results_dir / "metrics"
    summary = read_csv(m / "player_tracking_summary.csv")
    ball = read_csv(m / "ball_summary.csv")
    per_seq = read_csv(m / "ball_per_sequence.csv")
    bench = read_json(m / "runtime_benchmark.json")
    balljson = read_json(m / "ball.json")
    pub = read_csv(manifest.results_dir / "literature" / "sportsmot_test_published.csv")
    f = mlr.num
    n: dict[str, str] = {}

    def trk(group, a, ps="dev-tuned"):
        return next(r for r in summary if r["group"] == group and r["tracker"] == a and r["param_set"] == ps)

    def bl(group, method):
        return next(r for r in ball if r["group"] == group and r["method"] == method)

    names = {"sort": "SORT", "ocsort": "OCSORT", "bytetrack": "BYTE"}
    for a, k in names.items():
        sm, uv, de = trk("SportsMOT (test)", a), trk("UVY-Track (test)", a), trk("SportsMOT (test)", a, "default")
        for key, col in (("HOTA", "HOTA"), ("DETA", "DetA"), ("ASSA", "AssA"), ("MOTA", "MOTA"), ("IDF1", "IDF1"),
                         ("RECALL", "CLR_Re")):
            n[f"SM_{k}_{key}"] = f(sm[col])
        n[f"SM_{k}_IDSW"] = str(sm["IDSW"])
        n[f"SM_{k}_FP"] = f(sm["CLR_FP"])
        n[f"SM_{k}_FN"] = f(sm["CLR_FN"])
        n[f"UVY_{k}_OWTA"] = f(uv["OWTA"])
        n[f"UVY_{k}_DETRE"] = f(uv["DetRe"])
        n[f"UVY_{k}_IDSW"] = str(uv["IDSW"])
        n[f"DEF_{k}_HOTA"] = f(de["HOTA"])
        n[f"DEF_{k}_IDSW"] = str(de["IDSW"])
        n[f"DEF_{k}_MOTA"] = f(de["MOTA"])
    sm_b, sm_s = trk("SportsMOT (test)", "bytetrack"), trk("SportsMOT (test)", "sort")
    n["SM_EXTRA_FP"] = f(sm_b["CLR_FP"] - sm_s["CLR_FP"])
    n["SM_ASSA_GAP"] = f(abs(sm_s["AssA"] - sm_b["AssA"]))
    defh = [trk("SportsMOT (test)", a, "default")["HOTA"] for a in ALGORITHMS]
    n["DEF_HOTA_SPREAD"] = f(max(defh) - min(defh))

    top = max(pub, key=lambda r: r["HOTA"])
    pb = next(r for r in pub if r["method"] == "ByteTrack" and r["training_setup"] == "train")
    po = next(r for r in pub if r["method"] == "OC-SORT" and r["training_setup"] == "train")
    n.update({"TOP_NAME": top["method"], "TOP_SETUP": top["training_setup"], "TOP_HOTA": f(top["HOTA"]),
              "TOP_DETA": f(top["DetA"]), "TOP_ASSA": f(top["AssA"]),
              "TOP_ASSA_GAP": f(top["AssA"] - sm_b["AssA"]), "TOP_DETA_GAP": f(top["DetA"] - sm_b["DetA"]),
              "PUB_BYTE_HOTA": f(pb["HOTA"]), "PUB_BYTE_DETA": f(pb["DetA"]), "PUB_BYTE_ASSA": f(pb["AssA"]),
              "PUB_OC_HOTA": f(po["HOTA"]), "N_PUB": str(len(pub))})

    for g, k in (("InFactory", "INF"), ("UVY-Track", "UVY"), ("All ball sequences", "ALL")):
        for meth, mk in (("detection_only", "DET"), ("kalman", "KF"), ("kalman_motion", "MOT")):
            r = bl(g, meth)
            for key, col in (("P", "precision_dist"), ("R", "recall_dist"), ("F1", "f1_dist")):
                n[f"B_{k}_{mk}_{key}"] = f(r[col])
            n[f"B_{k}_{mk}_SEG"] = f(r["track_segments"])
        n[f"B_{k}_VIS"] = f(bl(g, "kalman_motion")["frames_ball_visible"])
    mot = bl("All ball sequences", "kalman_motion")
    n["B_PRED_P"] = f(mot["precision_predicted_dist"])
    n["B_DETP_P"] = f(mot["precision_detected_dist"])
    n["B_AP50"] = f(bl("All ball sequences", "detection_only")["ap50_candidates"])
    v04 = {r["method"]: r for r in per_seq if str(r["sequence"]) == "soccer_V04"}
    n["V04_KF_R"], n["V04_MOT_R"], n["V04_MOT_P"] = (f(v04["kalman"]["recall_dist"]), f(v04["kalman_motion"]["recall_dist"]),
                                                    f(v04["kalman_motion"]["precision_dist"]))
    k_rows = {(r["dataset"], str(r["sequence"])): r for r in per_seq if r["method"] == "kalman"}
    m_rows = {(r["dataset"], str(r["sequence"])): r for r in per_seq if r["method"] == "kalman_motion"}
    changed = [key for key in k_rows if any(k_rows[key][c] != m_rows[key][c] for c in ("tp_dist", "fp_dist", "fn_dist"))]
    n["MOT_CHANGED"] = str(len(changed))
    n["N_BALL_SEQ"] = str(len(k_rows))
    sc = balljson.get("uvy_ball_scoring", {})
    n["UVY_VIS0"] = f(sum(v["frames_ball_row_visibility_0"] for v in sc.values()))
    n["UVY_NOROW"] = f(sum(v["frames_without_ball_row"] for v in sc.values()))
    n["UVY_SCORED"] = f(sum(v["frames_scored"] for v in sc.values()))
    bsys = balljson["system"]
    n["BALL_HW"] = mlr.esc(str(bsys.get("cpu", "CPU")).replace("(R)", "").replace("(TM)", "")) + \
        f", Ultralytics~{bsys.get('ultralytics')}"

    stages = bench["stages"]
    total = bench["full_pipeline_ms_per_frame"]["bytetrack"]
    det = stages["detection"]["median_ms"]
    motion = next((v["median_ms"] for k, v in stages.items() if k.startswith("ball motion")), 0.0)
    byte_ms = next(v["median_ms"] for k, v in stages.items() if k.startswith("player filter") and "ByteTrack" in k)
    n.update({"RT_FPS": mlr.num(bench["full_pipeline_fps"]["bytetrack"], 1), "RT_DET_MS": f(det, 0),
              "RT_DET_SHARE": f(100 * det / total, 0), "RT_MOT_MS": f(motion, 0),
              "RT_MOT_SHARE": f(100 * motion / total, 0), "RT_BYTE_MS": mlr.num(byte_ms, 2),
              "RT_CPU": mlr.esc(str(bench["system"].get("cpu", "CPU")).replace("(R)", "").replace("(TM)", ""))})
    return n


def extra_tables(manifest) -> dict[str, str]:
    """Tables that the original paper generator does not build."""
    m = manifest.results_dir / "metrics"
    summary = read_csv(m / "player_tracking_summary.csv")
    pub = sorted(read_csv(manifest.results_dir / "literature" / "sportsmot_test_published.csv"), key=lambda r: r["HOTA"])
    f, row = mlr.num, mlr.row
    lines = []
    for r in pub:
        star = "" if r["training_setup"] == "train" else "$^{*}$"
        lines.append(row([mlr.esc(r["method"]) + star, f(r["HOTA"]), f(r["DetA"]), f(r["AssA"]), f(r["MOTA"]),
                          f(r["IDF1"])]))
    lines.append(r"\midrule")
    short = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack (proposed)"}
    for a in ALGORITHMS:
        r = next(x for x in summary if x["group"] == "SportsMOT (test)" and x["tracker"] == a
                 and x["param_set"] == "dev-tuned")
        lines.append(row([r"\textit{Ours: YOLO11n + " + short[a] + "}", f(r["HOTA"]), f(r["DetA"]), f(r["AssA"]),
                          f(r["MOTA"]), f(r["IDF1"])]))
    return {"TAB_SOTA": "\n".join(lines)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    ap.add_argument("--template", default="docs/report/report_body.template.tex")
    ap.add_argument("--out-dir", default="docs/report")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, src in FIGURES.items():
        src_path = manifest.results_dir / src
        if not src_path.is_file():
            raise SystemExit(f"figure source missing: {src_path} (run: python scripts/run_experiments.py plots)")
        shutil.copyfile(src_path, fig_dir / name)

    body = Path(args.template).read_text(encoding="utf-8")
    values = {**mlr.build_tables(manifest), **extra_tables(manifest), **numbers(manifest)}
    for key, value in values.items():
        body = body.replace(f"@@{key}@@", value)

    problems_total = 0
    for name, front in (("report_ieee", ieee_front()), ("report_single_column", single_front())):
        tex = front + "\n" + body
        path = out_dir / f"{name}.tex"
        path.write_text(tex, encoding="utf-8", newline="\n")
        problems = mlr.lint(tex, fig_dir)
        for line in problems:
            print(f"LINT {name}: {line}", file=sys.stderr)
        problems_total += len(problems)
        with zipfile.ZipFile(out_dir / f"{name}_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(path, f"{name}.tex")
            for fig in FIGURES:
                zf.write(fig_dir / fig, f"figures/{fig}")
        print(f"Wrote {path} ({len(tex.splitlines())} lines) and {out_dir / (name + '_overleaf.zip')}; "
              f"{len(problems)} lint problem(s)")
    return 1 if problems_total else 0


if __name__ == "__main__":
    sys.exit(main())
