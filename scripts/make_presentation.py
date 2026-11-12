"""Build presentation.pptx from the stored experiment results.

    python scripts/make_presentation.py

Every number on the slides is read from results/metrics/ at build time; figures
come from results/plots/ and results/demo/. Speaker notes for each slide are in
docs/PRESENTATION_NOTES.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from soccer_tracker.experiments import load_manifest
from soccer_tracker.experiments.common import read_csv, read_json
from soccer_tracker.tracking.factory import ALGORITHMS, SHORT_LABELS, TRACKER_LABELS

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x12, 0x8A, 0x60)
LIGHT = RGBColor(0xF3, 0xF2, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
W, H = Inches(13.333), Inches(7.5)


class Deck:
    def __init__(self, footer: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.footer = footer
        self.n = 0

    def slide(self, title: str, subtitle: str | None = None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        bar = s.shapes.add_shape(1, 0, 0, Inches(0.28), H)
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()
        self.text(s, title, Inches(0.6), Inches(0.3), Inches(12.2), Inches(0.8), size=30, bold=True)
        if subtitle:
            self.text(s, subtitle, Inches(0.6), Inches(1.02), Inches(12.2), Inches(0.5), size=15, color=INK2)
        self.text(s, self.footer, Inches(0.6), Inches(7.05), Inches(10.5), Inches(0.3), size=10, color=INK2)
        self.text(s, str(self.n), Inches(12.2), Inches(7.05), Inches(0.7), Inches(0.3), size=10, color=INK2,
                  align=PP_ALIGN.RIGHT)
        return s

    @staticmethod
    def text(s, text, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT):
        box = s.shapes.add_textbox(x, y, w, h)
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        return box

    @staticmethod
    def bullets(s, items, x, y, w, h, size=17, gap=7):
        box = s.shapes.add_textbox(x, y, w, h)
        tf = box.text_frame
        tf.word_wrap = True
        first = True
        for item in items:
            level, text = (item if isinstance(item, tuple) else (0, item))
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.space_after = Pt(gap)
            r = p.add_run()
            r.text = ("• " if level == 0 else "– ") + text
            r.font.size = Pt(size - 2 * level)
            r.font.color.rgb = INK if level == 0 else INK2
            p.level = level
        return box

    @staticmethod
    def image(s, path: Path, x, y, max_w, max_h):
        """Place an image inside a box, keeping its aspect ratio. Returns False if the file is missing."""
        if not Path(path).is_file():
            Deck.text(s, f"[figure not available: {Path(path).name}]", x, y, max_w, Inches(0.4), size=12, color=INK2)
            return False
        pic = s.shapes.add_picture(str(path), x, y)
        ratio = min(max_w / pic.width, max_h / pic.height)
        pic.width, pic.height = Emu(int(pic.width * ratio)), Emu(int(pic.height * ratio))
        pic.left = Emu(int(x + (max_w - pic.width) / 2))
        pic.top = Emu(int(y))
        return True

    @staticmethod
    def table(s, header, rows, x, y, w, row_h=Inches(0.42), size=12, col_widths=None, highlight_last=False):
        shape = s.shapes.add_table(len(rows) + 1, len(header), x, y, w, Emu(int(row_h * (len(rows) + 1))))
        tbl = shape.table
        if col_widths:
            total = sum(col_widths)
            for i, cw in enumerate(col_widths):
                tbl.columns[i].width = Emu(int(w * cw / total))
        for c, text in enumerate(header):
            cell = tbl.cell(0, c)
            cell.text = str(text)
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.bold, r.font.color.rgb = Pt(size), True, WHITE
        for ri, row in enumerate(rows, start=1):
            proposed = highlight_last and ri == len(rows)
            for c, v in enumerate(row):
                cell = tbl.cell(ri, c)
                cell.text = f"{v:.1f}" if isinstance(v, float) else str(v)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xE2, 0xF4, 0xEC) if proposed else (LIGHT if ri % 2 else WHITE)
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size, r.font.color.rgb, r.font.bold = Pt(size), INK, proposed
        return shape


def best(rows: dict[str, dict], key: str, higher: bool = True) -> str:
    """Name(s) of the tracker(s) with the best value of ``key``."""
    vals = {a: rows[a][key] for a in ALGORITHMS}
    target = max(vals.values()) if higher else min(vals.values())
    return " and ".join(SHORT_LABELS[a].replace(" (proposed)", "") for a in ALGORITHMS if vals[a] == target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    ap.add_argument("--output", default="presentation.pptx")
    ap.add_argument("--author", default="Team 8", help="Name(s) shown on the title slide")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    res = manifest.results_dir
    m, plots, demo = res / "metrics", res / "plots", res / "demo"
    summary = read_csv(m / "player_tracking_summary.csv")
    ball = read_csv(m / "ball_summary.csv")
    bench = read_json(m / "runtime_benchmark.json")
    sel = read_json(m / "player_selected_params.json")

    def trk(group: str, param_set: str = "dev-tuned") -> dict[str, dict]:
        return {a: next(r for r in summary if r["group"] == group and r["tracker"] == a and r["param_set"] == param_set)
                for a in ALGORITHMS}

    def bl(group: str, method: str) -> dict:
        return next(r for r in ball if r["group"] == group and r["method"] == method)

    sm_t, uvy_t = trk("SportsMOT (test)"), trk("UVY-Track (test)")
    seqs = manifest.all_sequences()
    n_frames = sum(s.info.num_frames for s in seqs)

    deck = Deck("Multi-Object Tracking for Sports Analytics  |  YOLO11n + ByteTrack + Kalman filter")

    # 1 ------------------------------------------------------------------ title
    s = deck.slide("Multi-Object Tracking for Sports Analytics", "Player and Ball Tracking")
    deck.text(s, "Proposed system: YOLO11n + ByteTrack (players) + Kalman filter with motion flights (ball)", Inches(0.6), Inches(2.0),
              Inches(12), Inches(0.6), size=24, bold=True, color=ACCENT)
    deck.bullets(s, [
        "Baselines: YOLO11n + SORT and YOLO11n + OC-SORT, on identical detections",
        "Datasets: SportsMOT, UVY-Track, InFactory Soccer Ball Tracking",
        f"{n_frames:,} labelled frames evaluated, all on a CPU",
    ], Inches(0.6), Inches(2.9), Inches(12), Inches(2.0), size=19)
    deck.text(s, f"Presented by: {args.author}", Inches(0.6), Inches(5.3), Inches(12), Inches(0.5), size=18, bold=True)
    deck.text(s, "Computer Vision, B.Tech Semester 7", Inches(0.6), Inches(5.8), Inches(12), Inches(0.5), size=16,
              color=INK2)

    # 2 ------------------------------------------------------- problem / objectives
    s = deck.slide("Problem statement and objectives")
    deck.text(s, "Problem", Inches(0.6), Inches(1.4), Inches(6), Inches(0.4), size=20, bold=True, color=ACCENT)
    deck.bullets(s, [
        "From one sports video, find every player in every frame and keep the same ID for the same player",
        "Find the ball: tiny (5-20 px), blurred, often hidden",
        "Say clearly whether a ball position was detected or only predicted",
        "Measure how well this works against baseline trackers",
    ], Inches(0.6), Inches(1.9), Inches(5.9), Inches(4.5), size=16)
    deck.text(s, "Objectives", Inches(6.9), Inches(1.4), Inches(6), Inches(0.4), size=20, bold=True, color=ACCENT)
    deck.bullets(s, [
        "Detect players and ball with a lightweight pretrained YOLO11n",
        "Track players with ByteTrack; track the ball with a Kalman filter and motion flights",
        "Export annotated video, tracking CSV and JSON summary",
        "Compare with SORT and OC-SORT on the same detections",
        "Evaluate on three public datasets with HOTA, MOTA, IDF1, precision / recall / F1, FPS",
        "Keep everything reproducible with one command",
    ], Inches(6.9), Inches(1.9), Inches(5.9), Inches(4.5), size=16)

    # 3 ------------------------------------------------------------------ datasets
    s = deck.slide("Datasets used", "Three public datasets, one loader each, annotations never mixed")
    counts = {d: (sum(1 for q in seqs if q.info.dataset == d), sum(q.info.num_frames for q in seqs if q.info.dataset == d))
              for d in ("SportsMOT", "UVY-Track", "InFactory")}
    deck.table(s, ["", "SportsMOT", "UVY-Track", "InFactory Ball Tracking"], [
        ["Footage", "Professional broadcast", "User-generated YouTube video", "Professional broadcast"],
        ["Annotation format", "MOT Challenge text", "MOT-style text + class column", "YOLO text per frame"],
        ["Annotated", "Players only", "Players, referees, goalkeepers, ball", "Ball only (+ not-visible frames)"],
        ["Track IDs", "Players", "People and ball", "None"],
        ["Used for", "Player tracking", "Player tracking + ball", "Ball detection only"],
        ["Subset evaluated", f"{counts['SportsMOT'][0]} sequences, {counts['SportsMOT'][1]:,} frames",
         f"{counts['UVY-Track'][0]} sequences, {counts['UVY-Track'][1]:,} frames",
         f"{counts['InFactory'][0]} clips, {counts['InFactory'][1]:,} frames"],
    ], Inches(0.6), Inches(1.7), Inches(12.1), row_h=Inches(0.5), size=13, col_widths=[1.3, 2, 2.3, 2.2])
    deck.bullets(s, [
        "One SportsMOT sequence is the only data used for tuning; it comes from a different source video than every test sequence",
        "Nothing is tuned on UVY-Track or InFactory",
        "About 0.73 GB downloaded instead of 11 GB: only the needed sequences are read from the remote archives",
    ], Inches(0.6), Inches(5.35), Inches(12.1), Inches(1.6), size=15)

    # 4 ---------------------------------------------------------------- architecture
    s = deck.slide("Proposed system architecture")
    deck.image(s, plots / "00_architecture.png", Inches(0.6), Inches(1.25), Inches(12.1), Inches(4.7))
    deck.bullets(s, [
        "One YOLO11n pass per frame serves players and ball; the trackers decide what to trust",
        "Hybrid tracking: multi-object association for players, a single-target motion model for the ball",
        "Motion flights: camera-compensated moving blobs linked into fast paths that start at the ball",
    ], Inches(0.6), Inches(6.0), Inches(12.1), Inches(1.0), size=15, gap=3)

    # 5 ----------------------------------------------------------------- methodology
    s = deck.slide("Methodology and tracking algorithms")
    deck.table(s, ["Tracker", "How detections are used", "Idea"], [
        ["SORT (baseline)", "Any detection can match a track", "Kalman prediction + one IoU matching stage"],
        ["OC-SORT (baseline)", "High-confidence detections only", "SORT + observation-centric correction after occlusion"],
        ["ByteTrack (proposed)", "High confidence first, then low confidence",
         "Second stage rescues tracks with low-score boxes"],
    ], Inches(0.6), Inches(1.4), Inches(12.1), row_h=Inches(0.5), size=13, col_widths=[1.4, 2.2, 3], highlight_last=True)
    deck.text(s, "Kalman ball tracker", Inches(0.6), Inches(3.65), Inches(6), Inches(0.4), size=18, bold=True,
              color=ACCENT)
    deck.bullets(s, [
        "Constant-velocity model, state [x, y, vx, vy]",
        "Gate around the prediction rejects false balls elsewhere",
        "Starts on a confident detection; reported after 2 detections",
        "Predicts at most 6 frames, labelled 'predicted'; not after a kick",
        "Follows a motion flight when the struck ball is lost by YOLO",
    ], Inches(0.6), Inches(4.1), Inches(5.9), Inches(2.8), size=15, gap=4)
    deck.text(s, "Fair comparison", Inches(6.9), Inches(3.65), Inches(6), Inches(0.4), size=18, bold=True, color=ACCENT)
    deck.bullets(s, [
        "YOLO11n detections cached once; all three trackers replay the same boxes",
        "Same frames, same ground truth, same evaluator",
        "Same 48-point parameter grid per tracker, selected by HOTA on the dev sequence",
        "Results also reported with untuned shared defaults",
    ], Inches(6.9), Inches(4.1), Inches(5.9), Inches(2.8), size=15, gap=4)

    # 6 ------------------------------------------------------------ baseline comparison
    r0 = sm_t["bytetrack"]
    s = deck.slide("Baseline comparison: player tracking",
                   "Identical YOLO11n detections for all three trackers; parameters chosen on a separate dev sequence")
    deck.text(s, f"SportsMOT test: {r0['sequences']} sequences, {r0['frames']:,} frames", Inches(0.6), Inches(1.5),
              Inches(12), Inches(0.35), size=14, bold=True, color=ACCENT)
    rows = [[TRACKER_LABELS[a]] + [sm_t[a][k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")] +
            [sm_t[a][k] for k in ("IDSW", "CLR_FP", "CLR_FN", "Frag")] for a in ALGORITHMS]
    deck.table(s, ["Method", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "ID sw.", "FP", "FN", "Frag"], rows, Inches(0.6),
               Inches(1.85), Inches(12.1), row_h=Inches(0.4), size=12,
               col_widths=[3.3, 1, 1, 1, 1, 1, 0.9, 1, 1, 0.9], highlight_last=True)
    ru = uvy_t["bytetrack"]
    deck.text(s, f"UVY-Track test: {ru['sequences']} sequences, {ru['frames']:,} frames "
                 "(only part of the people are labelled, so recall-based metrics are used)",
              Inches(0.6), Inches(3.55), Inches(12), Inches(0.35), size=14, bold=True, color=ACCENT)
    rows = [[TRACKER_LABELS[a]] + [uvy_t[a][k] for k in ("OWTA", "DetRe", "AssA")] +
            [uvy_t[a][k] for k in ("IDSW", "Frag", "CLR_FN")] for a in ALGORITHMS]
    deck.table(s, ["Method", "OWTA", "Det. recall", "AssA", "ID sw.", "Frag", "FN"], rows, Inches(0.6), Inches(3.9),
               Inches(7.6), row_h=Inches(0.4), size=12, col_widths=[3.3, 1, 1.2, 1, 0.9, 0.9, 0.9],
               highlight_last=True)
    deck.bullets(s, [
        f"SportsMOT: best HOTA {best(sm_t, 'HOTA')} ({max(sm_t[a]['HOTA'] for a in ALGORITHMS):.1f}); "
        f"ByteTrack {sm_t['bytetrack']['HOTA']:.1f}",
        f"Fewest ID switches on both datasets: {best(sm_t, 'IDSW', False)} "
        f"({sm_t['bytetrack']['IDSW']} and {uvy_t['bytetrack']['IDSW']})",
        f"Highest recall / fewest misses: {best(sm_t, 'CLR_FN', False)}",
        f"UVY-Track: best OWTA {best(uvy_t, 'OWTA')} ({max(uvy_t[a]['OWTA'] for a in ALGORITHMS):.1f})",
        "ByteTrack's extra false positives are real people the benchmark does not label",
    ], Inches(8.4), Inches(3.9), Inches(4.5), Inches(3.0), size=13, gap=4)
    deck.text(s, "Higher is better, except ID switches, FP, FN and Frag. Green row = proposed method.",
              Inches(0.6), Inches(5.75), Inches(7.6), Inches(0.6), size=12, color=INK2)

    # 7 --------------------------------------------------------------- metrics / ball
    s = deck.slide("Performance metrics and evaluation", "Ball: detection only vs Kalman vs Kalman + motion flights")
    rows = []
    for group in ("UVY-Track", "InFactory"):
        for method, name in (("detection_only", "Detection only"), ("kalman", "Kalman (YOLO only)"),
                             ("kalman_motion", "Kalman + motion (proposed)")):
            r = bl(group, method)
            rows.append([group, name, r["precision_dist"], r["recall_dist"], r["f1_dist"], r["coverage"],
                         r["frames_predicted"], r["track_segments"]])
    deck.table(s, ["Dataset", "Method", "Prec.", "Recall", "F1", "Cover.", "Predicted", "Segm."], rows,
               Inches(0.6), Inches(1.55), Inches(7.3), row_h=Inches(0.36), size=11,
               col_widths=[1.3, 2.9, 0.8, 0.9, 0.7, 0.9, 1.2, 0.9])
    deck.image(s, plots / "03_detection_prf.png", Inches(0.6), Inches(4.1), Inches(7.3), Inches(2.85))
    deck.text(s, "Metrics computed", Inches(8.2), Inches(1.55), Inches(4.6), Inches(0.4), size=18, bold=True, color=ACCENT)
    deck.bullets(s, [
        "Players: HOTA, DetA, AssA, MOTA, IDF1, ID switches, FP, FN, fragmentation, MT / ML",
        "Evaluator: trackers.eval (TrackEval-compatible)",
        "Ball: precision, recall, F1, AP@0.5, coverage, centre error, continuity",
        "Ball evaluator is our own, unit-tested; not an official benchmark tool",
        "Ball match: centre within one ball diameter of the truth",
        "No identity metrics on InFactory (it has no ball IDs)",
        "Runtime: ms per frame per stage, FPS",
    ], Inches(8.2), Inches(2.0), Inches(4.7), Inches(5.0), size=13, gap=4)

    # 8 ---------------------------------------------------------------- more graphs
    s = deck.slide("Graphs: association quality and identity stability")
    deck.image(s, plots / "02_deta_assa.png", Inches(0.5), Inches(1.3), Inches(6.2), Inches(2.8))
    deck.image(s, plots / "05_idsw_frag.png", Inches(6.9), Inches(1.3), Inches(6.2), Inches(2.8))
    deck.image(s, plots / "11_per_sequence_hota.png", Inches(0.5), Inches(4.25), Inches(6.2), Inches(2.7))
    deck.image(s, plots / "04_runtime.png", Inches(6.9), Inches(4.25), Inches(6.2), Inches(2.7))

    # 9 ---------------------------------------------------------------- visual results
    s = deck.slide("Visual results", "Annotated video, player trajectories, heatmap and ball trajectory")
    deck.image(s, demo / "uvy_soccer_V04_screenshot.png", Inches(0.5), Inches(1.6), Inches(6.2), Inches(2.65))
    deck.image(s, plots / "06_player_trajectories.png", Inches(6.9), Inches(1.6), Inches(6.2), Inches(2.65))
    deck.image(s, plots / "07_player_heatmap.png", Inches(0.5), Inches(4.35), Inches(6.2), Inches(2.6))
    deck.image(s, plots / "08_ball_trajectory.png", Inches(6.9), Inches(4.35), Inches(6.2), Inches(2.6))

    # 10 ------------------------------------------------------- limitations / future
    fps = bench["full_pipeline_fps"]["bytetrack"]
    det_ms = bench["stages"]["detection"]["median_ms"]
    s = deck.slide("Limitations and future improvements")
    deck.text(s, "Limitations", Inches(0.6), Inches(1.4), Inches(6), Inches(0.4), size=20, bold=True, color=ACCENT)
    deck.bullets(s, [
        "Generic COCO detector: it also finds referees, staff and coaches that the datasets do not label, which inflates false positives",
        "The proposed tracker is not best on every metric (tuned SORT has higher HOTA on SportsMOT)",
        "COCO 'sports ball' finds the ball in about a third of the frames; 40% of Kalman predictions are wrong",
        f"CPU only: {fps:.1f} FPS for the full pipeline; the detector takes {det_ms:.0f} ms per frame",
        "Motion flights only help on struck balls and must start where YOLO last saw the ball",
        "Small evaluation subset; not comparable with leaderboards",
        "No camera-motion compensation for players, re-identification or team labels",
    ], Inches(0.6), Inches(1.9), Inches(5.9), Inches(5.0), size=15)
    deck.text(s, "Future work", Inches(6.9), Inches(1.4), Inches(6), Inches(0.4), size=20, bold=True, color=ACCENT)
    deck.bullets(s, [
        "Soccer-specific detector weights for players and ball",
        "Tiled high-resolution inference for the ball",
        "Camera-motion compensation (BoT-SORT style)",
        "Team assignment from jersey colour; jersey numbers",
        "Pitch homography: positions in metres, speed, distance",
        "GPU / ONNX export for real-time speed",
        "Evaluation on the full validation splits",
    ], Inches(6.9), Inches(1.9), Inches(5.9), Inches(5.0), size=15)

    # 11 ---------------------------------------------------------------- conclusion
    ab_k, ab_d = bl("All ball sequences", "kalman_motion"), bl("All ball sequences", "detection_only")
    s = deck.slide("Conclusion")
    deck.bullets(s, [
        "A complete, tested pipeline: video in; annotated video, CSV and JSON out; runs on a CPU",
        f"Fair three-tracker comparison on {sm_t['bytetrack']['frames'] + uvy_t['bytetrack']['frames']:,} test frames "
        "of two datasets with identical detections",
        f"ByteTrack (proposed): fewest ID switches ({sm_t['bytetrack']['IDSW']} on SportsMOT vs "
        f"{sm_t['sort']['IDSW']} SORT, {sm_t['ocsort']['IDSW']} OC-SORT), highest recall, "
        f"best OWTA on UVY-Track ({uvy_t['bytetrack']['OWTA']:.1f})",
        f"Not best everywhere: tuned SORT has higher HOTA on SportsMOT ({sm_t['sort']['HOTA']:.1f} vs "
        f"{sm_t['bytetrack']['HOTA']:.1f}) because it starts fewer tracks on unlabelled people",
        f"Ball tracker (Kalman + motion flights) vs detection only: recall {ab_d['recall_dist']:.1f} to {ab_k['recall_dist']:.1f}, "
        f"F1 {ab_d['f1_dist']:.1f} to {ab_k['f1_dist']:.1f}, correct segments {ab_d['track_segments']} to "
        f"{ab_k['track_segments']}; precision drops {ab_d['precision_dist']:.1f} to {ab_k['precision_dist']:.1f}",
        "Predicted ball positions are always labelled as predictions: in the video, the CSV and the metrics",
        "The detector, not the tracker, limits accuracy and speed: a soccer-specific detector is the next step",
        "Reproducible with one command: python scripts/run_experiments.py all",
    ], Inches(0.6), Inches(1.45), Inches(12.1), Inches(5.5), size=17, gap=8)

    deck.prs.save(args.output)
    print(f"Wrote {args.output} ({deck.n} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
