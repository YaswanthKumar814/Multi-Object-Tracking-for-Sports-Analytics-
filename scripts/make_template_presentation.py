"""Build the project presentation on the institution's slide template.

    python scripts/make_template_presentation.py
    python scripts/make_template_presentation.py --template docs/presentation/Template.pptx --output Project_Presentation.pptx

The template's title slide (team table, advisor, logo) is kept and only its
title is filled in; its example content slide is replaced by the project slides,
which use the template's "Title and Content" layout. The slides follow the
evaluation rubric: problem and objectives, datasets, architecture, methodology,
evaluation protocol, results and the baseline comparison, comparison with the
state of the art, ball results, visual results, speed and code quality,
individual contributions, limitations, conclusion and references. Every number
is read from results/ at build time, and each slide carries short speaker notes.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from soccer_tracker.experiments import load_manifest
from soccer_tracker.experiments.common import read_csv, read_json
from soccer_tracker.tracking.factory import ALGORITHMS

CRIMSON = RGBColor(0xB8, 0x11, 0x4F)
CRIMSON_LIGHT = RGBColor(0xF8, 0xE3, 0xEB)
INK = RGBColor(0x1F, 0x1F, 0x1F)
INK2 = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF4, 0xF2, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BODY_FONT = "Times New Roman"
TITLE_FONT = "Georgia"
CONTENT_TOP, CONTENT_BOTTOM = 1.12, 6.85
SHORT = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack (proposed)"}


class Deck:
    def __init__(self, template: Path):
        self.prs = Presentation(str(template))
        self.layout = self.prs.slide_layouts.get_by_name("Title and Content") or self.prs.slide_layouts[1]
        # Remove the template's example content slide(s); keep the title slide.
        sld_ids = self.prs.slides._sldIdLst
        for sld_id in list(sld_ids)[1:]:
            self.prs.part.drop_rel(sld_id.rId)
            sld_ids.remove(sld_id)
        self.n = 1

    # -- slides -------------------------------------------------------------- #
    def title_slide(self, title: str, subtitle: str, notes: str) -> None:
        s = self.prs.slides[0]
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip() == "Project Title":
                p = sh.text_frame.paragraphs[0]
                p.runs[0].text = title
                for r in p.runs[1:]:
                    r.text = ""
                p.runs[0].font.size = Pt(32)
        box = s.shapes.add_textbox(Inches(0.39), Inches(1.22), Inches(12.56), Inches(0.6))
        para = box.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        r = para.add_run()
        r.text = subtitle
        r.font.size, r.font.name, r.font.color.rgb, r.font.italic = Pt(22), BODY_FONT, WHITE, True
        self._notes(s, notes)

    def slide(self, title: str, notes: str):
        s = self.prs.slides.add_slide(self.layout)
        self.n += 1
        for ph in list(s.placeholders):
            if ph.placeholder_format.idx == 0:
                ph.text_frame.text = title
                r = ph.text_frame.paragraphs[0].runs[0]
                r.font.name, r.font.size, r.font.color.rgb = TITLE_FONT, Pt(27), CRIMSON
                ph.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
            else:
                ph._element.getparent().remove(ph._element)  # we place our own content
        num = s.shapes.add_textbox(Inches(0.13), Inches(7.09), Inches(0.83), Inches(0.35))
        r = num.text_frame.paragraphs[0].add_run()
        r.text = str(self.n)
        r.font.size, r.font.name, r.font.color.rgb = Pt(10), BODY_FONT, WHITE
        self._notes(s, notes)
        return s

    @staticmethod
    def _notes(s, text: str) -> None:
        s.notes_slide.notes_text_frame.text = text

    # -- content ------------------------------------------------------------- #
    @staticmethod
    def text(s, x, y, w, h, paras, size=19, color=INK, bold=False, align=PP_ALIGN.LEFT, bullets=False, gap=6):
        """``paras``: list of strings or of [(text, bold), ...] runs."""
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.04)
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(gap)
            runs = [(para, bold)] if isinstance(para, str) else list(para)
            if bullets:
                runs = [("•  ", False)] + runs
            for t, b in runs:
                r = p.add_run()
                r.text = t
                r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), b, BODY_FONT, color
        return box

    @staticmethod
    def heading(s, x, y, w, text, size=21):
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.06), Inches(0.08), Inches(0.36))
        bar.fill.solid()
        bar.fill.fore_color.rgb = CRIMSON
        bar.line.fill.background()
        Deck.text(s, x + 0.16, y, w - 0.16, 0.5, [text], size=size, color=CRIMSON, bold=True)

    @staticmethod
    def image(s, path: Path, x, y, max_w, max_h) -> tuple[float, float]:
        pic = s.shapes.add_picture(str(path), Inches(x), Inches(y))
        ratio = min(Inches(max_w) / pic.width, Inches(max_h) / pic.height)
        pic.width, pic.height = Emu(int(pic.width * ratio)), Emu(int(pic.height * ratio))
        pic.left = Emu(int(Inches(x) + (Inches(max_w) - pic.width) / 2))
        return pic.width / 914400, pic.height / 914400

    @staticmethod
    def table(s, x, y, w, header, rows, col_widths, row_h=0.36, size=14, highlight=(), header_h=None):
        header_h = header_h or row_h
        shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w),
                                   Inches(header_h + row_h * len(rows)))
        tbl = shape.table
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
        for r_i in range(len(rows) + 1):
            tbl.rows[r_i].height = Inches(header_h if r_i == 0 else row_h)

        def fill(cell, value, bold, fg, bg):
            cell.text = value if isinstance(value, str) else (f"{value:.1f}" if isinstance(value, float) else str(value))
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            numeric = isinstance(value, (int, float))
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.RIGHT if numeric else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), bold, BODY_FONT, fg

        for c, h in enumerate(header):
            fill(tbl.cell(0, c), h, True, WHITE, CRIMSON)
        for i, row in enumerate(rows, start=1):
            hi = (i - 1) in highlight
            for c, v in enumerate(row):
                fill(tbl.cell(i, c), v, hi, INK, CRIMSON_LIGHT if hi else (LIGHT if i % 2 else WHITE))
        return row_h * (len(rows) + 1)

    @staticmethod
    def bar_chart(s, x, y, w, h, categories, series, colors, size=13):
        data = CategoryChartData()
        data.categories = categories
        for name, values in series.items():
            data.add_series(name, values)
        ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
        ch.font.size, ch.font.name = Pt(size), BODY_FONT
        ch.has_legend = True
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        plot = ch.plots[0]
        plot.gap_width, plot.overlap = 40, 0
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format, dl.number_format_is_linked = "0.0", False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(size - 2)
        for srs, color in zip(plot.series, colors):
            srs.format.fill.solid()
            srs.format.fill.fore_color.rgb = color
        ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 100
        ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE3, 0xDF)
        ch.category_axis.reverse_order = True

    @staticmethod
    def flow(s, x, y, w, h):
        """Architecture flow with native shapes (left to right)."""
        def box(bx, by, bw, bh, title, sub, fill, edge):
            shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(bx), Inches(by), Inches(bw), Inches(bh))
            shp.fill.solid()
            shp.fill.fore_color.rgb = fill
            shp.line.color.rgb = edge
            shp.line.width = Pt(1.75)
            tf = shp.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = Inches(0.05)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = title
            r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(15), True, BODY_FONT, INK
            if sub:
                p2 = tf.add_paragraph()
                p2.alignment = PP_ALIGN.CENTER
                r = p2.add_run()
                r.text = sub
                r.font.size, r.font.name, r.font.color.rgb = Pt(12), BODY_FONT, INK2

        def arrow(x1, y1, x2, y2):
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            c.line.color.rgb = INK2
            c.line.width = Pt(2)
            ln = c.line._get_or_add_ln()
            from lxml import etree
            tail = etree.SubElement(ln, qn("a:tailEnd"))
            tail.set("type", "triangle")

        grey, edge = WHITE, RGBColor(0xC9, 0xC8, 0xC3)
        bw, gap = 2.05, 0.5
        mid = y + h / 2
        lane_h = 1.05
        top_y, bot_y = y + 0.15, y + h - lane_h - 0.15
        box(x, mid - 0.6, bw, 1.2, "Video input", "frames, timestamps", grey, edge)
        x2 = x + bw + gap
        box(x2, mid - 0.6, bw, 1.2, "YOLO11n", "one pass: person + sports ball", RGBColor(0xEE, 0xF4, 0xFC),
            RGBColor(0x2A, 0x78, 0xD6))
        arrow(x + bw, mid, x2, mid)
        x3 = x2 + bw + gap
        box(x3, top_y, bw, lane_h, "Player filter", "size, aspect ratio", grey, edge)
        box(x3, bot_y, bw, lane_h, "Ball filter + motion", "size; moving blobs", grey, edge)
        arrow(x2 + bw, mid - 0.2, x3, top_y + lane_h / 2)
        arrow(x2 + bw, mid + 0.2, x3, bot_y + lane_h / 2)
        x4 = x3 + bw + gap
        box(x4, top_y, bw, lane_h, "ByteTrack", "persistent player IDs", CRIMSON_LIGHT, CRIMSON)
        box(x4, bot_y, bw, lane_h, "Kalman + flights", "detected / predicted ball", CRIMSON_LIGHT, CRIMSON)
        arrow(x3 + bw, top_y + lane_h / 2, x4, top_y + lane_h / 2)
        arrow(x3 + bw, bot_y + lane_h / 2, x4, bot_y + lane_h / 2)
        x5 = x4 + bw + gap
        box(x5, mid - 0.95, bw, 1.9, "Outputs", "annotated MP4, tracks CSV, JSON summary; evaluation and SQLite",
            grey, edge)
        arrow(x4 + bw, top_y + lane_h / 2, x5, mid - 0.3)
        arrow(x4 + bw, bot_y + lane_h / 2, x5, mid + 0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="docs/presentation/Template.pptx")
    ap.add_argument("--output", default="Project_Presentation.pptx")
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    res = manifest.results_dir
    m, plots, demo = res / "metrics", res / "plots", res / "demo"
    summary = read_csv(m / "player_tracking_summary.csv")
    ball = read_csv(m / "ball_summary.csv")
    per_seq = read_csv(m / "ball_per_sequence.csv")
    bench = read_json(m / "runtime_benchmark.json")
    pub = read_csv(res / "literature" / "sportsmot_test_published.csv")
    seqs = manifest.all_sequences()
    frames = {d: sum(q.info.num_frames for q in seqs if q.info.dataset == d) for d in ("SportsMOT", "UVY-Track", "InFactory")}
    nseq = {d: sum(1 for q in seqs if q.info.dataset == d) for d in frames}

    def trk(group, ps="dev-tuned"):
        return {a: next(r for r in summary if r["group"] == group and r["tracker"] == a and r["param_set"] == ps)
                for a in ALGORITHMS}

    def bl(group, method):
        return next(r for r in ball if r["group"] == group and r["method"] == method)

    def best(rows, key, higher=True):
        vals = {a: rows[a][key] for a in ALGORITHMS}
        t = max(vals.values()) if higher else min(vals.values())
        return " and ".join(SHORT[a].replace(" (proposed)", "") for a in ALGORITHMS if vals[a] == t)

    sm, uvy, smd = trk("SportsMOT (test)"), trk("UVY-Track (test)"), trk("SportsMOT (test)", "default")
    d = Deck(Path(args.template))

    # 1 ----------------------------------------------------------------- title
    d.title_slide("Multi-Object Tracking for Sports Analytics",
                  "Player and Ball Tracking with YOLO11n, ByteTrack and a Kalman Filter",
                  "Introduce the team and the topic: tracking every player with a stable ID and the ball in soccer "
                  "video, evaluated on three public datasets against two baseline trackers.")

    # 2 ----------------------------------------------------------------- problem
    s = d.slide("Problem Statement and Objectives",
                "Two signals drive sports analytics: player positions with stable identities, and the ball. "
                "Explain why soccer is hard (identical kits, occlusion, camera motion, tiny blurred ball) and list the "
                "objectives. Stress that predicted ball positions are never shown as detections.")
    d.heading(s, 0.6, CONTENT_TOP, 6.0, "Problem")
    d.text(s, 0.6, CONTENT_TOP + 0.55, 5.9, 5.0, [
        "From one soccer video, box every player in every frame and keep the same ID for the same player.",
        "Locate the ball, which is tiny (5-20 px), blurred when kicked and often hidden.",
        "Team-mates look identical, players occlude each other and the camera pans and zooms.",
        "Every ball position must say whether it was detected or only predicted.",
    ], bullets=True, size=19)
    d.heading(s, 6.9, CONTENT_TOP, 6.0, "Objectives")
    d.text(s, 6.9, CONTENT_TOP + 0.55, 5.9, 5.0, [
        "Detect players and the ball with a lightweight pretrained YOLO11n.",
        "Track players with ByteTrack; track the ball with a Kalman filter.",
        "Export an annotated video, a per-frame CSV and a JSON summary.",
        "Compare fairly with SORT and OC-SORT on identical detections.",
        "Evaluate on three public datasets and place the results against the state of the art.",
    ], bullets=True, size=19)

    # 3 ----------------------------------------------------------------- datasets
    s = d.slide("Datasets",
                "Three public datasets, each with its own loader; annotations are never mixed. Point out the label "
                "issues we found: UVY-Track labels only some people, and its visibility-0 ball boxes are not real ball "
                "positions, so those frames are not scored for the ball.")
    d.table(s, 0.6, CONTENT_TOP + 0.1, 12.1, ["Dataset", "Footage", "Labels used", "Used for", "Evaluated"], [
        ["SportsMOT (ICCV 2023)", "Professional broadcast", "Players, track IDs", "Player tracking",
         f"{nseq['SportsMOT']} sequences, {frames['SportsMOT']:,} frames (1 for tuning)"],
        ["UVY-Track (Zenodo)", "User-generated video", "People + ball, track IDs", "Players + ball",
         f"{nseq['UVY-Track']} sequences, {frames['UVY-Track']:,} frames"],
        ["InFactory ball tracking", "Professional broadcast", "Ball + 'not visible' frames", "Ball detection",
         f"{nseq['InFactory']} clips, {frames['InFactory']:,} frames"],
    ], [2.3, 2.1, 2.3, 1.6, 3.4], row_h=0.5, size=15)
    d.text(s, 0.6, CONTENT_TOP + 2.35, 12.1, 3.3, [
        [("Tuning vs test: ", True), ("one SportsMOT sequence is used only to choose tracker parameters; nothing is "
                                      "tuned on UVY-Track or InFactory.", False)],
        [("Partial labels: ", True), ("UVY-Track boxes only the people involved in the play, so false-positive "
                                      "metrics (MOTA, DetA) are not valid there; we use OWTA and recall.", False)],
        [("Ball labels: ", True), ("UVY-Track never marks the ball absent and its visibility-0 ball boxes sit on empty "
                                   "grass, so only frames with a valid ball box are scored; false alarms are measured "
                                   "on InFactory.", False)],
    ], bullets=True, size=18)

    # 4 ----------------------------------------------------------------- architecture
    s = d.slide("Proposed System Architecture",
                "Walk left to right: one YOLO11n pass serves both branches. Players go through a filter to ByteTrack; "
                "ball candidates and motion blobs go to the Kalman tracker. The crimson blocks are the proposed hybrid; "
                "the baselines swap only the ByteTrack block.")
    d.flow(s, 0.65, CONTENT_TOP + 0.2, 12.0, 3.6)
    d.text(s, 0.6, CONTENT_TOP + 4.05, 12.1, 1.7, [
        "Proposed hybrid: multi-object association for players (ByteTrack) and a single-target motion model for the ball "
        "(Kalman filter with motion flights).",
        "For the SORT and OC-SORT baselines only the ByteTrack block is replaced; detections and everything else stay "
        "identical.",
    ], bullets=True, size=18)

    # 5 ----------------------------------------------------------------- methodology
    s = d.slide("Methodology: Tracking Algorithms",
                "All three trackers predict with a Kalman filter and match by IoU; they differ in how they use "
                "detection confidence. ByteTrack rescues tracks with low-confidence boxes, which suits occluded "
                "players. The ball tracker is a constant-velocity Kalman filter with gating, confirmation, short "
                "predictions and re-acquisition; motion flights follow a kicked ball the detector misses.")
    d.heading(s, 0.6, CONTENT_TOP, 6.4, "Player trackers (same detections)")
    d.table(s, 0.6, CONTENT_TOP + 0.6, 6.3, ["Tracker", "Detections used", "Key idea"], [
        ["SORT", "Any detection", "Kalman prediction + one IoU matching stage"],
        ["OC-SORT", "Confident only", "Corrects the filter after an occlusion"],
        ["ByteTrack (proposed)", "Confident, then low", "Second stage keeps occluded players' tracks alive"],
    ], [1.6, 1.5, 3.2], row_h=0.8, size=16, highlight=(2,))
    d.heading(s, 7.25, CONTENT_TOP, 5.5, "Ball tracker")
    d.text(s, 7.25, CONTENT_TOP + 0.55, 5.5, 5.0, [
        "State [x, y, vx, vy], constant velocity, one step per frame.",
        "Gate around the prediction (80 px, growing to 160 px) rejects false balls.",
        "Reported after 2 detections; only a confident one (>= 0.3) starts a track.",
        "Coasts at most 6 frames, labelled 'predicted'; never right after a kick.",
        "Motion flights: camera-compensated frame differencing links a kicked, blurred ball into a flight.",
    ], bullets=True, size=19)

    # 6 ----------------------------------------------------------------- protocol & metrics
    s = d.slide("Evaluation Protocol and Performance Metrics",
                "Fairness: detections are cached once and replayed by every tracker; the same 48-setting grid is used "
                "per tracker on one separate sequence. Metrics: HOTA is the main one; OWTA replaces false-positive "
                "metrics on UVY-Track; ball metrics come from our own unit-tested evaluator.")
    d.heading(s, 0.6, CONTENT_TOP, 5.6, "Fair comparison")
    d.text(s, 0.6, CONTENT_TOP + 0.55, 5.6, 5.0, [
        "YOLO11n runs once per sequence; all trackers replay identical boxes.",
        "Same frames, same ground truth, same evaluator.",
        "Same 48-setting grid per tracker on one tuning sequence; best HOTA kept.",
        "Results also reported with one shared, untuned setting.",
    ], bullets=True, size=20)
    d.table(s, 6.5, CONTENT_TOP + 0.05, 6.25, ["Metric", "What it measures"], [
        ["HOTA = sqrt(DetA x AssA)", "Overall tracking accuracy (main metric)"],
        ["DetA / AssA", "Detection vs association accuracy"],
        ["MOTA, IDF1", "CLEAR-MOT accuracy; identity F1"],
        ["IDSW, Frag, FP, FN", "Identity switches, broken tracks, errors"],
        ["OWTA", "HOTA without the FP penalty (partial labels)"],
        ["Ball P / R / F1", "Centre within one ball diameter"],
        ["FPS, ms per frame", "Execution speed per pipeline stage"],
    ], [2.6, 3.65], row_h=0.62, size=15)

    # 7 ----------------------------------------------------------------- baseline comparison
    r0 = sm["bytetrack"]
    s = d.slide("Results: Baseline Comparison (SportsMOT)",
                f"Read the table: ByteTrack has the fewest ID switches ({sm['bytetrack']['IDSW']}) and misses the fewest "
                f"players, but tuned SORT has the best HOTA ({sm['sort']['HOTA']:.1f}). The gap is detection accuracy: "
                "ByteTrack keeps more low-confidence boxes, many of them unlabelled staff. With untuned shared settings "
                f"all three are within {max(smd[a]['HOTA'] for a in ALGORITHMS) - min(smd[a]['HOTA'] for a in ALGORITHMS):.1f} HOTA.")
    d.text(s, 0.6, CONTENT_TOP, 12.1, 0.4,
           [f"{r0['sequences']} test sequences, {r0['frames']:,} frames, identical YOLO11n detections, parameters tuned on a "
            "separate sequence"], size=16, color=INK2)
    d.table(s, 0.6, CONTENT_TOP + 0.45, 12.1, ["Method", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "ID sw.", "FP", "FN", "Frag"],
            [[f"YOLO11n + {SHORT[a]}", sm[a]["HOTA"], sm[a]["DetA"], sm[a]["AssA"], sm[a]["MOTA"], sm[a]["IDF1"],
              sm[a]["IDSW"], sm[a]["CLR_FP"], sm[a]["CLR_FN"], sm[a]["Frag"]] for a in ALGORITHMS],
            [3.2, 1, 1, 1, 1, 1, 1, 1, 1, 1], row_h=0.5, size=15, highlight=(2,))
    d.text(s, 0.6, CONTENT_TOP + 2.65, 12.1, 3.0, [
        f"Fewest identity switches: {best(sm, 'IDSW', False)} ({sm['bytetrack']['IDSW']} vs {sm['sort']['IDSW']} SORT, "
        f"{sm['ocsort']['IDSW']} OC-SORT); fewest misses: {best(sm, 'CLR_FN', False)}.",
        f"Best HOTA, MOTA and IDF1: {best(sm, 'HOTA')}. ByteTrack's association (AssA {sm['bytetrack']['AssA']:.1f}) is "
        f"close to SORT's ({sm['sort']['AssA']:.1f}); its extra false positives are unlabelled staff and referees.",
        f"OC-SORT is the weakest associator here (AssA {sm['ocsort']['AssA']:.1f}, {sm['ocsort']['IDSW']} ID switches).",
    ], bullets=True, size=18)

    # 8 ----------------------------------------------------------------- UVY + identity
    s = d.slide("Results: UVY-Track and Identity Stability",
                "UVY-Track labels only some people, so we compare with OWTA, recall and identity switches. ByteTrack "
                "leads on OWTA and has the fewest switches; the bar chart shows identity stability on both datasets.")
    d.text(s, 0.6, CONTENT_TOP, 6.0, 0.4, ["UVY-Track test (recall-based metrics)"], size=17, color=INK2, bold=True)
    d.table(s, 0.6, CONTENT_TOP + 0.45, 6.0, ["Tracker", "OWTA", "Det. recall", "AssA", "ID sw."],
            [[SHORT[a], uvy[a]["OWTA"], uvy[a]["DetRe"], uvy[a]["AssA"], uvy[a]["IDSW"]] for a in ALGORITHMS],
            [2.4, 1, 1.2, 1, 1], row_h=0.5, size=15, highlight=(2,))
    d.text(s, 0.6, CONTENT_TOP + 2.65, 6.0, 3.0, [
        f"Best OWTA: {best(uvy, 'OWTA')} ({max(uvy[a]['OWTA'] for a in ALGORITHMS):.1f}).",
        f"Fewest ID switches on both datasets: ByteTrack ({sm['bytetrack']['IDSW']} and {uvy['bytetrack']['IDSW']}).",
        "MOTA is negative on UVY-Track for every tracker: it counts real but unlabelled people as errors.",
    ], bullets=True, size=17)
    d.image(s, plots / "05_idsw_frag.png", 6.9, CONTENT_TOP + 0.2, 5.9, 5.3)

    # 9 ----------------------------------------------------------------- SOTA
    top = max(pub, key=lambda r: r["HOTA"])
    pb = next(r for r in pub if r["method"] == "ByteTrack" and r["training_setup"] == "train")
    po = next(r for r in pub if r["method"] == "OC-SORT" and r["training_setup"] == "train")
    b = sm["bytetrack"]
    s = d.slide("Comparison with the State of the Art",
                f"Published SportsMOT test results from the papers (SportsMOT ICCV 2023, Deep-EIoU, DiffMOT). The best, "
                f"{top['method']}, reaches HOTA {top['HOTA']:.1f}. Our association accuracy is within "
                f"{top['AssA'] - b['AssA']:.1f} points of it, but detection accuracy is {top['DetA'] - b['DetA']:.1f} points "
                "lower: the gap is the detector. Protocols differ, so this is context, not a leaderboard claim.")
    chosen = [(f"{top['method']} ({top['training_setup']})", top), ("OC-SORT (train)", po), ("ByteTrack (train)", pb)]
    chosen += [(f"Ours: {SHORT[a]}", sm[a]) for a in ("sort", "bytetrack", "ocsort")]
    d.bar_chart(s, 0.5, CONTENT_TOP - 0.05, 7.4, 5.75, [n for n, _ in chosen],
                {k: [r[k] for _, r in chosen] for k in ("HOTA", "DetA", "AssA")},
                [CRIMSON, RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34)])
    d.text(s, 8.1, CONTENT_TOP + 0.1, 4.75, 5.6, [
        [("Gap is detection: ", True), (f"best published AssA {top['AssA']:.1f} vs ours {b['AssA']:.1f}; DetA "
                                         f"{top['DetA']:.1f} vs ours {b['DetA']:.1f}.", False)],
        [("Same HOTA, opposite reasons: ", True), (f"published ByteTrack {pb['HOTA']:.1f} (DetA {pb['DetA']:.1f}) vs ours "
                                                   f"{b['HOTA']:.1f} (DetA {b['DetA']:.1f}).", False)],
        [("Ranking depends on the detector: ", True), (f"published OC-SORT {po['HOTA']:.1f} > ByteTrack "
                                                       f"{pb['HOTA']:.1f}; with ours ByteTrack is ahead.", False)],
        [("Not directly comparable: ", True), ("published detectors are trained on SportsMOT and scored on its full test "
                                               "set; ours is COCO-pretrained, on 3 football validation sequences.", False)],
    ], bullets=True, size=16)

    # 10 ---------------------------------------------------------------- ball
    s = d.slide("Results: Ball Detection and Tracking",
                "On the held-out InFactory clips the tracker beats detection alone; on UVY-Track it does not, so pooled "
                "F1 is slightly lower than detection alone. The clear gain is continuity. Motion flights change only "
                "the free-kick clip, where recall goes to almost 100 percent, but that clip was used to develop them.")
    rows, hl = [], []
    for g, label in (("InFactory", "InFactory (held out)"), ("UVY-Track", "UVY-Track"), ("All ball sequences", "All")):
        for meth, name in (("detection_only", "Detection only"), ("kalman", "Kalman"), ("kalman_motion", "Kalman + motion")):
            r = bl(g, meth)
            if meth == "kalman_motion":
                hl.append(len(rows))
            rows.append([label if meth == "detection_only" else "", name, r["precision_dist"], r["recall_dist"],
                         r["f1_dist"]])
    d.table(s, 0.6, CONTENT_TOP, 6.6, ["Data", "Method", "Prec.", "Recall", "F1"], rows, [2.0, 1.9, 0.9, 0.9, 0.9],
            row_h=0.53, size=14, highlight=tuple(hl))
    inf_d, inf_m = bl("InFactory", "detection_only"), bl("InFactory", "kalman_motion")
    uvy_d, uvy_m = bl("UVY-Track", "detection_only"), bl("UVY-Track", "kalman_motion")
    all_d, all_m = bl("All ball sequences", "detection_only"), bl("All ball sequences", "kalman_motion")
    v04 = {r["method"]: r for r in per_seq if str(r["sequence"]) == "soccer_V04"}
    d.text(s, 7.5, CONTENT_TOP, 5.3, 5.7, [
        f"Held-out InFactory: F1 {inf_m['f1_dist']:.1f} vs {inf_d['f1_dist']:.1f}, precision {inf_m['precision_dist']:.1f} "
        f"vs {inf_d['precision_dist']:.1f}.",
        f"UVY-Track: detection alone is better (F1 {uvy_d['f1_dist']:.1f} vs {uvy_m['f1_dist']:.1f}); pooled "
        f"{all_d['f1_dist']:.1f} vs {all_m['f1_dist']:.1f}.",
        f"Continuity: {all_d['track_segments']} short correct pieces merge into {all_m['track_segments']}.",
        f"Free kick (soccer_V04): motion flights raise recall {v04['kalman']['recall_dist']:.1f} -> "
        f"{v04['kalman_motion']['recall_dist']:.1f}% at {v04['kalman_motion']['precision_dist']:.0f}% precision "
        "(development clip).",
        "Detector bottleneck: COCO 'sports ball' finds the ball in well under half of the visible frames.",
    ], bullets=True, size=17)

    # 11 ---------------------------------------------------------------- visuals
    s = d.slide("Visual Results",
                "Show the annotated output: IDs with trails for players, the ball marker and its path; predicted ball "
                "positions are orange and labelled. The right figure plots the ball against ground truth on the free "
                "kick. If possible, play outputs/uvy_soccer_V04_tracked_h264.mp4.")
    w, h = d.image(s, demo / "uvy_soccer_V04_screenshot.png", 0.5, CONTENT_TOP + 0.15, 7.1, 4.8)
    d.text(s, 0.5, CONTENT_TOP + 0.25 + h, 7.1, 0.9, ["Annotated output (UVY-Track soccer_V04): player IDs and trails, "
                                                     "ball and path; predictions in orange."], size=14, color=INK2)
    d.image(s, plots / "08_ball_trajectory.png", 7.8, CONTENT_TOP, 5.1, 5.7)

    # 12 ---------------------------------------------------------------- speed & code
    stages = bench["stages"]
    total = bench["full_pipeline_ms_per_frame"]["bytetrack"]
    rows = []
    for k, v in sorted(stages.items(), key=lambda kv: -kv[1]["median_ms"]):
        if k.startswith("player filter") and "ByteTrack" not in k:
            continue
        rows.append([k.replace(" (JPEG decode)", "").replace(" (mp4v)", "").replace(" (proposed)", "")
                      .replace("ball motion: camera compensation + flight linking", "ball motion stage"),
                     round(v["median_ms"], 2), round(100 * v["median_ms"] / total, 1)])
    s = d.slide("Execution Speed and Code Quality",
                f"{bench['full_pipeline_fps']['bytetrack']:.1f} frames per second on the CPU; the detector dominates the "
                "time, ByteTrack costs under a millisecond. Code: modular package, 122 automated tests, every result "
                "reproduced by one command, and the report tables are generated from the stored metrics.")
    d.heading(s, 0.6, CONTENT_TOP, 6.0, f"Runtime ({bench['system'].get('cpu', 'CPU')}, CPU)")
    d.table(s, 0.6, CONTENT_TOP + 0.6, 6.2, ["Stage", "ms / frame", "Share %"], rows, [3.6, 1.3, 1.3], row_h=0.44,
            size=14)
    d.text(s, 0.6, CONTENT_TOP + 0.75 + 0.44 * (len(rows) + 1), 6.2, 0.6,
           [f"Full pipeline: {bench['full_pipeline_fps']['bytetrack']:.1f} FPS"], size=17, bold=True)
    d.heading(s, 7.2, CONTENT_TOP, 5.6, "Code and reproducibility")
    d.text(s, 7.2, CONTENT_TOP + 0.55, 5.6, 5.1, [
        "Modular Python package: detection, tracking, datasets, evaluation, experiments.",
        "122 automated tests (unit, end-to-end smoke test, real-model tests); all pass.",
        "One command reproduces every experiment; data subsets fixed in a manifest.",
        "Command-line app: video in; annotated MP4, CSV and JSON out.",
        "Tables, poster and reports are generated from the stored results.",
    ], bullets=True, size=18)

    # 13 ---------------------------------------------------------------- contributions
    s = d.slide("Individual Contributions",
                "Each member owns specific components and answers for them in the viva; details and evidence commits "
                "are in CONTRIBUTIONS.md in the repository.")
    d.table(s, 0.5, CONTENT_TOP + 0.05, 12.3, ["Member", "Main responsibilities", "Answers for in the viva"], [
        ["Pulapa Yaswanth Kumar\nBL.EN.U4CSE23245",
         "Core application (video I/O, YOLO detection and filters, ByteTrack player tracking, first Kalman ball tracker, "
         "visualisation, CSV/JSON export, CLI); SORT/OC-SORT baselines and fair comparison protocol; dataset loaders; "
         "evaluation framework; experiments, plots, database; tests; first full experiment run",
         "ByteTrack vs SORT vs OC-SORT; Kalman filter; HOTA, MOTA, IDF1, OWTA; datasets"],
        ["Ravva Siddhartha\nBL.EN.U4CSE23246",
         "Motion flights for the ball in flight (camera-motion compensation, frame differencing, flight linking); "
         "hardening of the Kalman ball tracker; motion flights in the experiments; final regenerated results and "
         "result videos; IEEE paper (related work, references)",
         "Motion stage and flight rules; final results; ball-tracking changes"],
    ], [2.3, 6.8, 3.2], row_h=1.75, size=15, header_h=0.5)
    d.text(s, 0.5, CONTENT_TOP + 4.3, 12.3, 0.5, ["An AI coding assistant was used during development; commits that "
                                                   "carry its co-author trailer are listed in CONTRIBUTIONS.md."],
           size=14, color=INK2)

    # 14 ---------------------------------------------------------------- limitations
    s = d.slide("Limitations and Future Work",
                "Be open about the limits: the generic detector dominates the error, the proposed tracker is not best "
                "on every metric, the evaluation is small, and the ball motion model cannot follow direction changes.")
    d.heading(s, 0.6, CONTENT_TOP, 6.0, "Limitations")
    d.text(s, 0.6, CONTENT_TOP + 0.55, 5.9, 5.1, [
        "Generic COCO detector: finds unlabelled people and misses most small balls.",
        "ByteTrack is not best on every metric (tuned SORT has higher HOTA).",
        "Small evaluation: 6 tracking test sequences, 1 tuning sequence; no confidence intervals.",
        "Constant-velocity ball model cannot follow a direction change during a gap.",
        "No camera-motion compensation for players, re-identification or team labels.",
    ], bullets=True, size=20)
    d.heading(s, 6.9, CONTENT_TOP, 6.0, "Future work")
    d.text(s, 6.9, CONTENT_TOP + 0.55, 5.9, 5.1, [
        "Soccer-trained detector with player, official and ball classes.",
        "Tiled high-resolution inference for the ball.",
        "Motion model for direction changes; offline smoothing.",
        "Camera-motion compensation and re-identification (BoT-SORT style).",
        "Team assignment, pitch calibration, GPU for real time.",
    ], bullets=True, size=20)

    # 15 ---------------------------------------------------------------- conclusion
    s = d.slide("Conclusion",
                "Summarise the three findings: ByteTrack gives the most stable identities, the gap to the state of the "
                "art is the detector, and the Kalman ball tracker gives continuity and helps on held-out data but not "
                "everywhere. Close with the main next step.")
    d.text(s, 0.8, CONTENT_TOP + 0.2, 11.8, 5.4, [
        "A complete, tested pipeline: video in; annotated video, CSV and JSON out; runs on a CPU.",
        f"Fair comparison on identical detections: ByteTrack gives the most stable identities "
        f"({sm['bytetrack']['IDSW']} ID switches on SportsMOT vs {sm['sort']['IDSW']} and {sm['ocsort']['IDSW']}) and the "
        f"best OWTA on UVY-Track; tuned SORT has the best HOTA on SportsMOT.",
        f"Against published results our association is near the state of the art; the gap "
        f"(HOTA {b['HOTA']:.1f} vs {top['HOTA']:.1f}) is in detection.",
        f"Ball: the Kalman tracker helps on held-out InFactory (F1 {inf_m['f1_dist']:.1f} vs {inf_d['f1_dist']:.1f}) and "
        "makes the track far more continuous; predicted positions are always labelled.",
        "Next step: a detector trained on soccer data.",
    ], bullets=True, size=20, gap=10)

    # 16 ---------------------------------------------------------------- references
    s = d.slide("References", "Key references; the full list is in the report.")
    d.text(s, 0.6, CONTENT_TOP + 0.05, 12.1, 5.7, [
        "[1] Y. Zhang et al., \"ByteTrack: Multi-object tracking by associating every detection box,\" ECCV, 2022.",
        "[2] A. Bewley et al., \"Simple online and realtime tracking,\" ICIP, 2016.",
        "[3] J. Cao et al., \"Observation-centric SORT,\" CVPR, 2023.",
        "[4] Y. Cui et al., \"SportsMOT: A large multi-object tracking dataset in multiple sports scenes,\" ICCV, 2023.",
        "[5] H.-W. Huang et al., \"Iterative scale-up ExpansionIoU and deep features association for MOT in sports,\" "
        "arXiv:2306.13074, 2023.",
        "[6] W. Lv et al., \"DiffMOT: A real-time diffusion-based multiple object tracker,\" arXiv:2403.02075, 2024.",
        "[7] E. Alencar and R. de Freitas, \"UVY: A dataset of sport user-generated videos,\" Zenodo 21303900.",
        "[8] V. Constantinou et al., \"Soccer ball tracking dataset,\" Infactory, Hugging Face, 2026.",
        "[9] J. Luiten et al., \"HOTA: A higher order metric for evaluating multi-object tracking,\" IJCV, 2021.",
        "[10] R. E. Kalman, \"A new approach to linear filtering and prediction problems,\" J. Basic Eng., 1960.",
        "[11] G. Jocher and J. Qiu, \"Ultralytics YOLO11,\" 2024.",
    ], size=15, gap=4)

    d.prs.save(args.output)
    print(f"Wrote {args.output} ({len(d.prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
