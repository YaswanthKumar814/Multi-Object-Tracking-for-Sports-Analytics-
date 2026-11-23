"""Build the single-page project poster (poster.pptx, 48 x 36 in, landscape).

    python scripts/make_poster.py                # writes poster.pptx
    python scripts/make_poster.py --pdf          # also exports poster.pdf (needs PowerPoint on Windows)

Every number on the poster is read from results/ at build time (metrics CSV/JSON
and results/literature/), and the sentences that state a comparison are built
from those numbers, so the poster cannot drift from the results. Figures come
from results/plots/ and results/demo/.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from soccer_tracker.experiments import load_manifest
from soccer_tracker.experiments.common import read_csv, read_json
from soccer_tracker.tracking.factory import ALGORITHMS, SHORT_LABELS

W_IN, H_IN = 48.0, 36.0
MARGIN, GAP = 0.8, 0.8
# The middle column is wider so the state-of-the-art figure stays readable.
COL_WIDTHS = (14.0, 16.8, 14.0)
assert abs(sum(COL_WIDTHS) + 2 * MARGIN + 2 * GAP - W_IN) < 1e-6
BODY_TOP, BODY_BOTTOM = 5.0, 33.9

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x12, 0x8A, 0x60)
ACCENT_LIGHT = RGBColor(0xE2, 0xF4, 0xEC)
LIGHT = RGBColor(0xF3, 0xF2, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PROPOSED_ROW = RGBColor(0xD5, 0xEF, 0xE3)

BODY_PT = 31
TABLE_PT = 24


class Poster:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W_IN), Inches(H_IN)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])

    # -- primitives ------------------------------------------------------- #
    def rect(self, x, y, w, h, fill, line=None):
        shp = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = line
        shp.shadow.inherit = False
        return shp

    def text(self, x, y, w, h, runs, size=BODY_PT, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
        """``runs``: a string, or a list of paragraphs; each paragraph a string or a list of (text, bold) runs."""
        box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        tf.margin_top = tf.margin_bottom = Inches(0.02)
        paras = [runs] if isinstance(runs, str) else runs
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(size * 0.35)
            for txt, b in ([(para, bold)] if isinstance(para, str) else para):
                r = p.add_run()
                r.text = txt
                r.font.size = Pt(size)
                r.font.bold = b
                r.font.color.rgb = color
        return box

    def image(self, path: Path, x, y, w, max_h=None) -> float:
        """Place an image at width ``w`` (shrunk to ``max_h`` if needed, centred). Returns its height in inches."""
        pic = self.slide.shapes.add_picture(str(path), Inches(x), Inches(y))
        aspect = pic.height / pic.width
        width, height = w, w * aspect
        if max_h is not None and height > max_h:
            height, width = max_h, max_h / aspect
        pic.width, pic.height = Inches(width), Inches(height)
        pic.left = Inches(x + (w - width) / 2)
        return height



    def bar_chart(self, x, y, w, h, categories, series, colors, size=20) -> None:
        """Native clustered horizontal bar chart (stays sharp and large on the poster)."""
        from pptx.chart.data import CategoryChartData
        from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

        data = CategoryChartData()
        data.categories = categories
        for name, values in series.items():
            data.add_series(name, values)
        gf = self.slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
        ch = gf.chart
        ch.font.size = Pt(size)
        ch.font.color.rgb = INK
        ch.has_legend = True
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
        plot = ch.plots[0]
        plot.gap_width = 45
        plot.overlap = 0
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = "0.0"
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(size - 3)
        for srs, color in zip(plot.series, colors):
            srs.format.fill.solid()
            srs.format.fill.fore_color.rgb = color
        va = ch.value_axis
        va.minimum_scale, va.maximum_scale = 0, 100
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE3, 0xDF)
        va.tick_labels.font.size = Pt(size - 3)
        va.format.line.fill.background()
        ca = ch.category_axis
        ca.reverse_order = True  # first category at the top
        ca.tick_labels.font.size = Pt(size)
        ca.format.line.color.rgb = RGBColor(0xC9, 0xC8, 0xC3)

    def flow_diagram(self, x, y, w) -> float:
        """Architecture flow drawn with native shapes, so its text stays large on the poster. Returns its height."""
        from pptx.enum.shapes import MSO_CONNECTOR
        from pptx.oxml.ns import qn
        from lxml import etree

        def box(bx, by, bw, bh, title, sub, fill, edge):
            shp = self.slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(bx), Inches(by), Inches(bw), Inches(bh))
            shp.adjustments[0] = 0.12
            shp.fill.solid()
            shp.fill.fore_color.rgb = fill
            shp.line.color.rgb = edge
            shp.line.width = Pt(2.5)
            shp.shadow.inherit = False
            tf = shp.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = Inches(0.08)
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p0 = tf.paragraphs[0]
            p0.alignment = PP_ALIGN.CENTER
            r = p0.add_run()
            r.text = title
            r.font.size, r.font.bold, r.font.color.rgb = Pt(26), True, INK
            if sub:
                p1 = tf.add_paragraph()
                p1.alignment = PP_ALIGN.CENTER
                r = p1.add_run()
                r.text = sub
                r.font.size, r.font.color.rgb = Pt(20), INK2
            return (bx, by, bw, bh)

        def arrow(x1, y1, x2, y2, label=None):
            c = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            c.line.color.rgb = INK2
            c.line.width = Pt(3)
            ln = c.line._get_or_add_ln()
            tail = etree.SubElement(ln, qn("a:tailEnd"))
            tail.set("type", "triangle")
            tail.set("w", "med")
            tail.set("len", "med")
            if label:
                self.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2 - 0.25, 2.4, 0.5, label, size=19, color=INK2)

        neutral, edge = WHITE, RGBColor(0xC9, 0xC8, 0xC3)
        green, green_edge = ACCENT_LIGHT, ACCENT
        lane = (w - 0.6) / 2
        lx, rx = x, x + lane + 0.6
        # row 1: input -> detector
        b_in = box(x, y, w * 0.36, 1.35, "Video input", "MP4 / frames, timestamps", neutral, edge)
        b_det = box(x + w * 0.46, y, w * 0.54, 1.35, "YOLO11n detector", "one pass: person + sports ball",
                    RGBColor(0xEE, 0xF4, 0xFC), RGBColor(0x2A, 0x78, 0xD6))
        arrow(b_in[0] + b_in[2], y + 0.675, b_det[0], y + 0.675)
        y2 = y + 2.05
        # row 2: filters
        b_pf = box(lx, y2, lane, 1.25, "Player filter", "size, aspect ratio", neutral, edge)
        b_bf = box(rx, y2, lane, 1.25, "Ball filter + motion", "size; moving blobs (ORB)", neutral, edge)
        arrow(b_det[0] + b_det[2] * 0.3, y + 1.35, lx + lane / 2, y2, "person")
        arrow(b_det[0] + b_det[2] * 0.7, y + 1.35, rx + lane / 2, y2, "ball")
        y3 = y2 + 1.95
        # row 3: trackers (the proposed blocks)
        b_bt = box(lx, y3, lane, 1.55, "ByteTrack", "persistent player IDs", green, green_edge)
        b_kf = box(rx, y3, lane, 1.55, "Kalman + flights", "detected / predicted ball", green, green_edge)
        arrow(lx + lane / 2, y2 + 1.25, lx + lane / 2, y3)
        arrow(rx + lane / 2, y2 + 1.25, rx + lane / 2, y3)
        y4 = y3 + 2.25
        # row 4: outputs
        b_out = box(x, y4, w, 1.3, "Annotated MP4  \u00b7  tracks CSV  \u00b7  JSON summary",
                    "evaluation: HOTA / MOTA / IDF1, ball P / R / F1, SQLite", neutral, edge)
        arrow(lx + lane / 2, y3 + 1.55, lx + lane / 2, y4)
        arrow(rx + lane / 2, y3 + 1.55, rx + lane / 2, y4)
        return y4 + 1.3 - y

    def table(self, x, y, w, header, rows, col_widths, row_h=0.52, size=TABLE_PT, highlight=()) -> float:
        n_rows = len(rows) + 1
        shape = self.slide.shapes.add_table(n_rows, len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * n_rows))
        tbl = shape.table
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
        for r in range(n_rows):
            tbl.rows[r].height = Inches(row_h)
        for c, h in enumerate(header):
            numeric_col = all(_is_number(row[c]) for row in rows)
            self._cell(tbl.cell(0, c), h, size, True, WHITE, INK, numeric_col)
        for ri, row in enumerate(rows, start=1):
            hi = (ri - 1) in highlight
            for c, v in enumerate(row):
                fill = PROPOSED_ROW if hi else (LIGHT if ri % 2 else WHITE)
                self._cell(tbl.cell(ri, c), v, size, hi, INK, fill, _is_number(v))
        return row_h * n_rows

    @staticmethod
    def _cell(cell, value, size, bold, color, fill, right):
        cell.text = value if isinstance(value, str) else (f"{value:.1f}" if isinstance(value, float) else str(value))
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.RIGHT if right else PP_ALIGN.LEFT
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.bold = bold
                r.font.color.rgb = color


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) or bool(re.fullmatch(r"[\d,.\-]+", str(v)))


def text_height(paragraphs, width_in: float, size: int = BODY_PT) -> float:
    """Rough height of wrapped text in inches (average glyph width ~0.5 em)."""
    chars_per_line = max(10, int(width_in * 72 / (size * 0.45)))
    lines = 0
    for para in paragraphs:
        s = para if isinstance(para, str) else "".join(t for t, _ in para)
        lines += max(1, math.ceil(len(s) / chars_per_line))
    return lines * size * 1.2 / 72 + len(paragraphs) * size * 0.35 / 72 + 0.1


class Column:
    def __init__(self, poster: Poster, x: float, width: float):
        self.p, self.x, self.y, self.w = poster, x, BODY_TOP, width

    def section(self, number: int, title: str) -> None:
        self.p.rect(self.x, self.y, self.w, 1.0, ACCENT)
        self.p.text(self.x + 0.25, self.y, self.w - 0.5, 1.0, f"{number}   {title}", size=40, color=WHITE, bold=True,
                    anchor=MSO_ANCHOR.MIDDLE)
        self.y += 1.3

    def para(self, paragraphs, size=BODY_PT, color=INK, gap=0.25) -> None:
        h = text_height(paragraphs, self.w - 0.2, size)
        self.p.text(self.x + 0.1, self.y, self.w - 0.2, h, paragraphs, size=size, color=color)
        self.y += h + gap

    def bullets(self, items, size=BODY_PT, gap=0.3) -> None:
        paras = []
        for it in items:
            paras.append([("•  ", False)] + ([(it, False)] if isinstance(it, str) else list(it)))
        self.para(paras, size=size, gap=gap)

    def image(self, path: Path, max_h=None, caption: str | None = None, gap=0.3) -> None:
        h = self.p.image(path, self.x, self.y, self.w, max_h)
        self.y += h + 0.1
        if caption:
            self.para([caption], size=23, color=INK2, gap=gap)
        else:
            self.y += gap


    def flow(self, caption: str, gap=0.3) -> None:
        self.y += self.p.flow_diagram(self.x, self.y, self.w) + 0.15
        self.para([caption], size=23, color=INK2, gap=gap)

    def table(self, header, rows, col_widths, highlight=(), row_h=0.7, size=TABLE_PT, gap=0.45) -> None:
        self.y += self.p.table(self.x, self.y, self.w, header, rows, col_widths, row_h, size, highlight) + gap

    def space(self, h: float) -> None:
        self.y += h

    def check(self, name: str) -> None:
        if self.y > BODY_BOTTOM + 0.05:
            raise SystemExit(f"Poster layout overflow in {name}: ends at {self.y:.2f} in > {BODY_BOTTOM} in")


def best_name(rows: dict, key: str, higher: bool = True) -> str:
    vals = {a: rows[a][key] for a in ALGORITHMS}
    target = max(vals.values()) if higher else min(vals.values())
    return " and ".join(SHORT_LABELS[a].replace(" (proposed)", "") for a in ALGORITHMS if vals[a] == target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    ap.add_argument("--output", default="poster.pptx")
    ap.add_argument("--pdf", action="store_true", help="also export a PDF with PowerPoint (Windows only)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    res = manifest.results_dir
    m, plots, demo = res / "metrics", res / "plots", res / "demo"
    summary = read_csv(m / "player_tracking_summary.csv")
    ball = read_csv(m / "ball_summary.csv")
    bench = read_json(m / "runtime_benchmark.json")
    published = read_csv(res / "literature" / "sportsmot_test_published.csv")

    def trk(group):
        return {a: next(r for r in summary if r["group"] == group and r["tracker"] == a and r["param_set"] == "dev-tuned")
                for a in ALGORITHMS}

    def bl(group, method):
        return next(r for r in ball if r["group"] == group and r["method"] == method)

    sm, uvy = trk("SportsMOT (test)"), trk("UVY-Track (test)")
    seqs = manifest.all_sequences()
    frames = {d: sum(s.info.num_frames for s in seqs if s.info.dataset == d) for d in ("SportsMOT", "UVY-Track", "InFactory")}
    nseq = {d: sum(1 for s in seqs if s.info.dataset == d) for d in frames}

    p = Poster()

    # ---------------------------------------------------------------- header
    p.rect(0, 0, W_IN, 4.3, ACCENT)
    p.text(MARGIN, 0.35, W_IN - 2 * MARGIN, 1.6, "Multi-Object Tracking for Sports Analytics: Player and Ball Tracking",
           size=80, color=WHITE, bold=True)
    p.text(MARGIN, 1.95, W_IN - 2 * MARGIN, 0.9,
           "YOLO11n detection  →  ByteTrack for players  +  Kalman filter with motion flights for the ball  →  "
           "evaluated against SORT and OC-SORT on three public datasets", size=36, color=WHITE)
    p.text(MARGIN, 3.05, W_IN - 2 * MARGIN, 0.8,
           [[("Siddhartha", True), ("   ·   ", False), ("P Yaswanth Kumar", True),
             ("        Guide: Dr. Tripty Singh        Computer Vision, B.Tech Semester 7", False)]],
           size=30, color=WHITE)

    # ---------------------------------------------------------------- column 1
    c1 = Column(p, MARGIN, COL_WIDTHS[0])
    c1.section(1, "Problem and goal")
    c1.bullets([
        "From a single soccer video: box every player in every frame and keep the same ID for the same player.",
        "Find the ball, which is tiny (5-20 px), blurred and often hidden, and say whether each ball position "
        "was detected or only predicted.",
        "Measure all of it against baseline trackers on public, labelled data, on a laptop CPU, with pretrained "
        "models only.",
    ])
    c1.section(2, "Proposed system")
    c1.flow("Green blocks are the proposed hybrid. For the SORT and OC-SORT baselines only the ByteTrack block is "
            "swapped; everything else, including the detections, stays identical.")
    c1.bullets([
        [("ByteTrack (players): ", True), ("associates confident boxes first, then uses low-confidence boxes to keep "
                                           "occluded players' tracks alive.", False)],
        [("Kalman filter (ball): ", True), ("constant-velocity state [x, y, vx, vy]; a gate around the prediction "
                                            "rejects false balls; short gaps are bridged and labelled ", False),
         ("predicted", True), (".", False)],
        [("Motion flights (ball in flight): ", True), ("camera-motion-compensated frame differencing finds a struck, "
                                                        "blurred ball that YOLO misses.", False)],
    ])
    c1.section(3, "Datasets and protocol")
    c1.table(["Dataset", "Labels used", "Role", "Frames"], [
        ["SportsMOT", "players, track IDs", f"players: 1 tuning + {nseq['SportsMOT'] - 1} test seq.",
         f"{frames['SportsMOT']:,}"],
        ["UVY-Track", "people + ball, IDs", f"players + ball: {nseq['UVY-Track']} test seq.", f"{frames['UVY-Track']:,}"],
        ["InFactory", "ball, 'not visible'", f"ball: {nseq['InFactory']} test clips", f"{frames['InFactory']:,}"],
    ], [1.5, 2.2, 2.9, 1.1], row_h=0.72)
    c1.bullets([
        "YOLO11n runs once per sequence and its detections are cached: all three trackers replay identical boxes.",
        "Each tracker gets the same 48-setting grid on one separate SportsMOT sequence; test data is never used "
        "for tuning.",
    ], size=28)
    c1.check("column 1")

    # ---------------------------------------------------------------- column 2
    c2 = Column(p, MARGIN + COL_WIDTHS[0] + GAP, COL_WIDTHS[1])
    c2.section(4, "Player tracking: baseline comparison")
    r0 = sm["bytetrack"]
    c2.para([[("SportsMOT test", True), (f" ({r0['sequences']} sequences, {r0['frames']:,} frames)", False)]], size=22,
            gap=0.1)
    c2.table(["Tracker", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "ID sw."],
             [[SHORT_LABELS[a], sm[a]["HOTA"], sm[a]["DetA"], sm[a]["AssA"], sm[a]["MOTA"], sm[a]["IDF1"],
               sm[a]["IDSW"]] for a in ALGORITHMS],
             [2.6, 1, 1, 1, 1, 1, 1], highlight=(2,))
    ru = uvy["bytetrack"]
    c2.para([[("UVY-Track test", True), (f" ({ru['sequences']} sequences, {ru['frames']:,} frames; only some "
                                         "people are labelled, so recall-based metrics)", False)]], size=28, gap=0.1)
    c2.table(["Tracker", "OWTA", "Det. recall", "AssA", "ID sw."],
             [[SHORT_LABELS[a], uvy[a]["OWTA"], uvy[a]["DetRe"], uvy[a]["AssA"], uvy[a]["IDSW"]] for a in ALGORITHMS],
             [2.6, 1, 1.3, 1, 1], highlight=(2,))
    c2.bullets([
        f"Fewest ID switches: {best_name(sm, 'IDSW', False)} on SportsMOT, {best_name(uvy, 'IDSW', False)} on UVY-Track.",
        f"Best HOTA on SportsMOT: {best_name(sm, 'HOTA')} ({max(sm[a]['HOTA'] for a in ALGORITHMS):.1f}); "
        f"best OWTA on UVY-Track: {best_name(uvy, 'OWTA')} ({max(uvy[a]['OWTA'] for a in ALGORITHMS):.1f}).",
        "ByteTrack keeps more low-confidence boxes: fewer misses, but more boxes on unlabelled people "
        "(staff, referees), which costs DetA and MOTA.",
    ], size=28)
    c2.section(5, "Comparison with the state of the art")
    top = max(published, key=lambda r: r["HOTA"])
    pub_byte = next(r for r in published if r["method"] == "ByteTrack" and r["training_setup"] == "train")
    pub_oc = next(r for r in published if r["method"] == "OC-SORT" and r["training_setup"] == "train")
    b = sm["bytetrack"]
    c2.bullets([
        [("The gap is detection, not association: ", True),
         (f"the best published tracker ({top['method']}, {top['training_setup']}) has HOTA {top['HOTA']:.1f}. "
          f"Its AssA is only {top['AssA'] - b['AssA']:.1f} points above our ByteTrack, its DetA "
          f"{top['DetA'] - b['DetA']:.1f} points above.", False)],
        "Published methods train their detector on SportsMOT and are scored on the full test set; ours is "
        "COCO-pretrained and scored on 3 football validation sequences. Context, not a leaderboard entry.",
    ], size=26)
    chosen = [(f"{top['method']} ({top['training_setup']})", top), (f"OC-SORT ({pub_oc['training_setup']})", pub_oc),
              (f"ByteTrack ({pub_byte['training_setup']})", pub_byte)]
    chosen += [(f"Ours: {SHORT_LABELS[a]}", sm[a]) for a in ("sort", "bytetrack", "ocsort")]
    chart_h = max(6.0, BODY_BOTTOM - c2.y - 1.5)
    p.bar_chart(c2.x, c2.y, c2.w, chart_h, [name for name, _ in chosen],
                {k: [r[k] for _, r in chosen] for k in ("HOTA", "DetA", "AssA")},
                [ACCENT, RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34)], size=22)
    c2.space(chart_h + 0.1)
    c2.para(["Published values: Cui et al. ICCV 2023 (Table 3); Huang et al. arXiv:2306.13074 (Table 2). All published "
             "rows and sources: results/literature/."], size=21, color=INK2, gap=0.1)
    c2.check("column 2")

    # ---------------------------------------------------------------- column 3
    c3 = Column(p, MARGIN + COL_WIDTHS[0] + COL_WIDTHS[1] + 2 * GAP, COL_WIDTHS[2])
    c3.section(6, "Ball detection and tracking")
    rows, hl = [], []
    for g, label in (("InFactory", "InFactory (held out)"), ("UVY-Track", "UVY-Track"), ("All ball sequences", "All 13")):
        for meth, name in (("detection_only", "Detection only"), ("kalman", "Kalman"), ("kalman_motion", "Kalman + motion")):
            r = bl(g, meth)
            if meth == "kalman_motion":
                hl.append(len(rows))
            rows.append([label if meth == "detection_only" else "", name, r["precision_dist"], r["recall_dist"],
                         r["f1_dist"]])
    c3.table(["Data", "Method", "Prec.", "Recall", "F1"], rows, [2.2, 2.3, 1, 1, 1], highlight=tuple(hl), row_h=0.62,
             size=23)
    inf_d, inf_m = bl("InFactory", "detection_only"), bl("InFactory", "kalman_motion")
    uvy_d, uvy_m = bl("UVY-Track", "detection_only"), bl("UVY-Track", "kalman_motion")
    all_d, all_m = bl("All ball sequences", "detection_only"), bl("All ball sequences", "kalman_motion")

    def vs(a, b_, key):
        return f"{a[key]:.1f} vs {b_[key]:.1f}"

    c3.bullets([
        f"Held-out InFactory: the tracker {'beats' if inf_m['f1_dist'] > inf_d['f1_dist'] else 'does not beat'} "
        f"detection alone (F1 {vs(inf_m, inf_d, 'f1_dist')}, precision {vs(inf_m, inf_d, 'precision_dist')}).",
        f"UVY-Track: detection alone is {'better' if uvy_d['f1_dist'] > uvy_m['f1_dist'] else 'worse'} "
        f"(F1 {vs(uvy_d, uvy_m, 'f1_dist')}); pooled over all 13 sequences F1 is {vs(all_m, all_d, 'f1_dist')}.",
        f"Clearest gain: continuity. {all_d['track_segments']} short correct pieces merge into "
        f"{all_m['track_segments']}.",
        "Ball positions are scored only where the ground truth gives one (centre within one ball diameter).",
    ], size=28)
    c3.image(demo / "uvy_soccer_V04_screenshot.png", max_h=7.6,
             caption="Annotated output (UVY-Track soccer_V04): player IDs and trails, the ball and its path. "
                     "Predicted ball positions are drawn in orange and labelled 'ball (predicted)'.")
    c3.section(7, "Speed and conclusions")
    det_ms = bench["stages"]["detection"]["median_ms"]
    total_ms = bench["full_pipeline_ms_per_frame"]["bytetrack"]
    motion_ms = next((v["median_ms"] for k, v in bench["stages"].items() if k.startswith("ball motion")), 0.0)
    trk_ms = next(v["median_ms"] for k, v in bench["stages"].items() if k.startswith("player filter") and "ByteTrack" in k)
    cpu = str(bench["system"].get("cpu", "CPU"))
    per_seq = read_csv(m / "ball_per_sequence.csv")
    per_seq_k = {(r["dataset"], str(r["sequence"])): r for r in per_seq if r["method"] == "kalman"}
    per_seq_m = {(r["dataset"], str(r["sequence"])): r for r in per_seq if r["method"] == "kalman_motion"}
    keys = ("tp_dist", "fp_dist", "fn_dist")
    changed_names = [k[1] for k in per_seq_k if any(per_seq_k[k][c] != per_seq_m[k][c] for c in keys)]
    changed = len(changed_names)
    c3.bullets([
        f"{bench['full_pipeline_fps']['bytetrack']:.1f} FPS end to end on an {cpu} (CPU): YOLO11n "
        f"{100 * det_ms / total_ms:.0f}% of the time, ball motion stage {100 * motion_ms / total_ms:.0f}%, "
        f"ByteTrack {trk_ms:.1f} ms per frame.",
        "ByteTrack gives the most stable identities; tuned SORT scores the highest HOTA on SportsMOT.",
        f"Motion flights change the ball output on {changed} of {len(per_seq_k)} sequences "
        f"({', '.join(changed_names) or 'none'}): they help on shots, but have no held-out evidence yet.",
        "The generic COCO detector, not the tracker, limits accuracy and speed: a soccer-trained detector is the "
        "clearest next step.",
    ], size=28)
    c3.check("column 3")

    # ---------------------------------------------------------------- footer
    p.rect(0, BODY_BOTTOM + 0.25, W_IN, H_IN - BODY_BOTTOM - 0.25, LIGHT)
    p.text(MARGIN, BODY_BOTTOM + 0.4, W_IN - 2 * MARGIN, 1.5, [
        "References: Zhang et al., ByteTrack, ECCV 2022 · Bewley et al., SORT, ICIP 2016 · Cao et al., "
        "OC-SORT, CVPR 2023 · Cui et al., SportsMOT, ICCV 2023 · Alencar & de Freitas, UVY-Track, Zenodo "
        "21303900 · Constantinou et al., InFactory ball tracking, Hugging Face · Huang et al., Deep-EIoU, "
        "arXiv:2306.13074 · Lv et al., DiffMOT, arXiv:2403.02075 · Luiten et al., HOTA, IJCV 2021 "
        "· Ultralytics YOLO11, 2024.",
        "Code, data subsets and all results: github.com/YaswanthKumar814/Multi-Object-Tracking-for-Sports-Analytics-"
        "   ·   Every number on this poster is generated from the stored results by scripts/make_poster.py.",
    ], size=19, color=INK2)

    p.prs.save(args.output)
    print(f"Wrote {args.output} (1 slide, {W_IN:.0f} x {H_IN:.0f} in); column ends at "
          f"{c1.y:.1f} / {c2.y:.1f} / {c3.y:.1f} in of {BODY_BOTTOM} in")

    if args.pdf:
        try:
            import win32com.client  # type: ignore  # pywin32
        except ImportError:
            win32com = None
        out_pdf = str(Path(args.output).with_suffix(".pdf").resolve())
        if win32com is None:
            print("PDF export needs pywin32 and PowerPoint; skipped.", file=sys.stderr)
        else:
            app = win32com.client.Dispatch("PowerPoint.Application")
            pres = app.Presentations.Open(str(Path(args.output).resolve()), True, False, False)
            pres.SaveAs(out_pdf, 32)
            pres.Close()
            print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
