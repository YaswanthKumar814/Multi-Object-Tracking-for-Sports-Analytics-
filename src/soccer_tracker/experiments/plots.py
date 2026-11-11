"""Figures for the report, drawn only from stored experiment results.

Every figure reads the CSV/JSON files written by the ``players`` / ``ball`` /
``models`` stages (and, for the overlays, the saved tracks and the original
frames). Nothing is hard-coded: a figure whose inputs are missing is skipped
and reported, never filled with placeholder values.

Colours: one fixed colour per tracker everywhere (SORT blue, OC-SORT orange,
ByteTrack aqua) from a palette checked for colour-vision-deficiency
separation; bars carry value labels so identity never depends on colour alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from ..tracking.factory import ALGORITHMS, SHORT_LABELS
from .common import read_csv, read_json, write_json
from .manifest import Manifest

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
TRACKER_COLORS = {"sort": "#2a78d6", "ocsort": "#eb6834", "bytetrack": "#1baf7a"}
BASELINE_GRAY = "#8a8983"
PROPOSED = "#1baf7a"
DATASET_COLORS = {"SportsMOT": "#6250d6", "UVY-Track": "#e87ba4", "InFactory": "#eda100"}
BALL_COLORS = {"detected": "#2a78d6", "predicted": "#eb6834", "lost": "#c9c8c3", "gt": "#52514e"}
MAIN_SET = "dev-tuned"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 170,
    })
    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _grouped_bars(ax, categories: list[str], series: list[tuple[str, str, list[float]]], ylabel: str,
                  fmt: str = "{:.1f}", ylim: tuple[float, float] | None = None) -> None:
    """Grouped vertical bars with a value label on each bar."""
    n = len(series)
    width = min(0.8 / n, 0.26)
    x = np.arange(len(categories))
    top = max((v for _, _, vals in series for v in vals if v is not None), default=1.0)
    bottom = min((v for _, _, vals in series for v in vals if v is not None), default=0.0)
    for i, (label, color, values) in enumerate(series):
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        vals = [0 if v is None else v for v in values]
        ax.bar(pos, vals, width, color=color, label=label, linewidth=0)
        for px, v in zip(pos, values):
            if v is None:
                continue
            ax.text(px, v + (0.012 * (top - min(bottom, 0)) if v >= 0 else -0.012 * (top - bottom)), fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(min(0, bottom * 1.15), top * 1.14 if top > 0 else 1)
    if bottom < 0:
        ax.axhline(0, color=INK2, linewidth=0.8)
    ax.tick_params(axis="x", length=0)


def _summary(manifest: Manifest) -> list[dict]:
    return read_csv(manifest.results_dir / "metrics" / "player_tracking_summary.csv")


def _row(rows: list[dict], group: str, tracker: str, param_set: str = MAIN_SET) -> dict:
    return next(r for r in rows if r["group"] == group and r["tracker"] == tracker and r["param_set"] == param_set)


# UVY-Track labels only some of the people in each clip (see DATASETS.md), so false positives
# cannot be measured there. Its panels therefore show recall- and association-based metrics.
UVY_NOTE = "UVY-Track labels only part of the people:\nFP-based metrics (MOTA, DetA) are not meaningful"


def _tracker_panels(manifest: Manifest, panels: list[tuple[str, list[str]]], title: str, ylabel: str,
                    fmt: str = "{:.1f}"):
    """One panel per (group, metrics); the three trackers side by side in each."""
    plt = _plt()
    rows = _summary(manifest)
    panels = [(g, ms) for g, ms in panels if any(r["group"] == g for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.0), squeeze=False)
    for ax, (group, metrics) in zip(axes[0], panels):
        series = [(SHORT_LABELS[a], TRACKER_COLORS[a], [_row(rows, group, a)[m] for m in metrics]) for a in ALGORITHMS]
        _grouped_bars(ax, metrics, series, ylabel, fmt, (0, 100))
        r = _row(rows, group, "bytetrack")
        ax.set_title(f"{group}: {r['sequences']} sequences, {r['frames']} frames")
        ax.set_xlabel("Metric")
        if group.startswith("UVY"):
            ax.text(0.5, 0.97, UVY_NOTE, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK2)
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.5 * len(panels) + 0.02 * (len(panels) - 1), -0.18),
                      ncol=3)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    return fig


# --------------------------------------------------------------------------- #
# 1-2, 5: tracker comparison
# --------------------------------------------------------------------------- #
def plot_hota_mota_idf1(manifest: Manifest, out: Path) -> None:
    fig = _tracker_panels(manifest, [("SportsMOT (test)", ["HOTA", "MOTA", "IDF1"]),
                                     ("UVY-Track (test)", ["OWTA", "AssA", "DetRe"])],
                          "Player tracking accuracy: SORT vs OC-SORT vs ByteTrack (same YOLO11n detections)",
                          "Score (%), higher is better")
    _save(fig, out)


def plot_deta_assa(manifest: Manifest, out: Path) -> None:
    fig = _tracker_panels(manifest, [("SportsMOT (test)", ["DetA", "AssA"]),
                                     ("UVY-Track (test)", ["DetRe", "AssA"])],
                          "Detection vs association: DetA / AssA (SportsMOT), detection recall / AssA (UVY-Track)",
                          "Score (%), higher is better")
    _save(fig, out)


def plot_idsw_frag(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    rows = _summary(manifest)
    groups = [g for g in ("SportsMOT (test)", "UVY-Track (test)") if any(r["group"] == g for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for ax, metric, name in zip(axes, ["IDSW", "Frag"], ["Identity switches", "Track fragmentations"]):
        series = [(SHORT_LABELS[a], TRACKER_COLORS[a], [_row(rows, g, a)[metric] for g in groups]) for a in ALGORITHMS]
        _grouped_bars(ax, [g.replace(" (test)", "") for g in groups], series, f"{name} (count), lower is better",
                      "{:.0f}")
        ax.set_title(name)
        ax.set_xlabel("Dataset (test sequences)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.16), ncol=3)
    fig.suptitle("Identity stability: ID switches and track fragmentation", fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out)


def plot_per_sequence_hota(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    rows = [r for r in read_csv(manifest.results_dir / "metrics" / "player_tracking_per_sequence.csv")
            if r["param_set"] == MAIN_SET and r["role"] == "test"]
    panels = [(ds, metric) for ds, metric in (("SportsMOT", "HOTA"), ("UVY-Track", "OWTA"))
              if any(r["dataset"] == ds for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.2), squeeze=False)
    for ax, (ds, metric) in zip(axes[0], panels):
        seqs = list(dict.fromkeys(r["sequence"] for r in rows if r["dataset"] == ds))
        series = []
        for a in ALGORITHMS:
            vals = [next(r[metric] for r in rows if r["tracker"] == a and r["dataset"] == ds and r["sequence"] == q)
                    for q in seqs]
            series.append((SHORT_LABELS[a], TRACKER_COLORS[a], vals))
        _grouped_bars(ax, [q.replace("v_", "") for q in seqs], series, f"{metric} (%), higher is better",
                      ylim=(0, 100))
        ax.set_xlabel("Test sequence")
        ax.set_title(f"{ds}: {metric} per test sequence")
        ax.tick_params(axis="x", labelsize=8)
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.5 * len(panels) + 0.02 * (len(panels) - 1), -0.18),
                      ncol=3)
    fig.suptitle("Per-sequence results (HOTA on SportsMOT, OWTA on the partially labelled UVY-Track)",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 3: precision / recall / F1
# --------------------------------------------------------------------------- #
def plot_detection_prf(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    mdir = manifest.results_dir / "metrics"
    det = [r for r in read_csv(mdir / "player_detection.csv") if r["role"] == "test"]
    ball = read_csv(mdir / "ball_summary.csv")
    ball_groups = [g for g in ("UVY-Track", "InFactory") if any(r["group"] == g for r in ball)]
    fig, axes = plt.subplots(1, 1 + len(ball_groups), figsize=(5.0 * (1 + len(ball_groups)), 4.1), squeeze=False)
    cats = ["Precision", "Recall", "F1"]

    ax = axes[0][0]
    series = []
    for ds in dict.fromkeys(r["dataset"] for r in det):
        rs = [r for r in det if r["dataset"] == ds]
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        if ds == "UVY-Track":  # partially labelled: only recall can be measured
            series.append((ds + " (recall only)", DATASET_COLORS[ds], [None, 100 * rc, None]))
        else:
            series.append((ds, DATASET_COLORS[ds], [100 * p, 100 * rc, 100 * 2 * p * rc / max(p + rc, 1e-9)]))
    _grouped_bars(ax, cats, series, "Score (%), higher is better", ylim=(0, 112))
    ax.set_title("Players: YOLO11n detector\n(confidence >= 0.5, IoU >= 0.5)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)

    for ax, group in zip(axes[0][1:], ball_groups):
        series = []
        for method, label, color in (("detection_only", "Detection only", BASELINE_GRAY),
                                     ("kalman", "Kalman tracker (proposed)", PROPOSED)):
            r = next(x for x in ball if x["group"] == group and x["method"] == method)
            series.append((label, color, [r["precision_dist"], r["recall_dist"], r["f1_dist"]]))
        _grouped_bars(ax, cats, series, "Score (%), higher is better", ylim=(0, 112))
        r = next(x for x in ball if x["group"] == group and x["method"] == "kalman")
        ax.set_title(f"Ball on {group}: {r['frames_ball_visible']} visible-ball frames\n"
                     "(centre within one ball diameter)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.suptitle("Detection quality: precision, recall and F1 for players and ball", fontsize=12,
                 fontweight="bold", y=1.04)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 4: runtime
# --------------------------------------------------------------------------- #
def plot_runtime(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    rows = _summary(manifest)
    group = "All test sequences"
    bench = read_json(manifest.results_dir / "metrics" / "runtime_benchmark.json")
    info = bench["system"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 0.55})
    labels = [SHORT_LABELS[a] for a in ALGORITHMS]
    colors = [TRACKER_COLORS[a] for a in ALGORITHMS]

    vals = [_row(rows, group, a)["tracker_ms_per_frame"] for a in ALGORITHMS]
    bars = axes[0].bar(labels, vals, 0.5, color=colors, linewidth=0)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Latency (ms per frame), lower is better")
    axes[0].set_xlabel("Tracker")
    axes[0].set_title("Tracking stage only\n(mean over all test sequences)")
    axes[0].set_ylim(0, max(vals) * 1.18)

    vals = [bench["full_pipeline_fps"][a] for a in ALGORITHMS]
    bars = axes[1].bar(labels, vals, 0.5, color=colors, linewidth=0)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Throughput (frames per second), higher is better")
    axes[1].set_xlabel("Tracker used in the full pipeline")
    axes[1].set_title("Full pipeline: read + detect + track\n+ ball + draw + encode")
    axes[1].set_ylim(0, max(vals) * 1.18)
    for ax in axes[:2]:
        ax.tick_params(axis="x", length=0, labelsize=8.5)

    stages = [(k, v["median_ms"]) for k, v in bench["stages"].items()
              if not (k.startswith("player filter") and "ByteTrack" not in k)]
    stages.sort(key=lambda kv: kv[1])
    names = [k for k, _ in stages]
    ms = [v for _, v in stages]
    axes[2].barh(names, ms, 0.55, color=["#2a78d6" if k == "detection" else BASELINE_GRAY for k in names], linewidth=0)
    for i, v in enumerate(ms):
        axes[2].text(v + max(ms) * 0.01, i, f"{v:.2f} ms", va="center", fontsize=8.5)
    axes[2].set_xlim(0, max(ms) * 1.2)
    axes[2].set_xlabel("Median time per frame (ms)")
    axes[2].set_ylabel("")
    axes[2].grid(axis="y", visible=False)
    axes[2].grid(axis="x", visible=True)
    axes[2].set_title("Where the time goes: %d frames of %s" % (bench["frames"], bench["sequence"].split("/")[1]))
    fig.suptitle("Runtime on %s (device: %s, %s px input)" % (info.get("cpu", "CPU"), info.get("device", "cpu"),
                                                             bench["detector"].split("_")[-1]),
                 fontsize=12, fontweight="bold", y=1.03)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 6-7: player trajectories and heatmap
# --------------------------------------------------------------------------- #
def _proposed_tracks(manifest: Manifest, seq) -> np.ndarray:
    path = manifest.results_dir / "tracks" / MAIN_SET / "bytetrack" / f"{seq.info.key}.txt"
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    return rows[:, :6]


def _overlay_sequence(manifest: Manifest):
    """Sequence used for the image overlays: the first SportsMOT test sequence, else first UVY one."""
    seqs = manifest.sportsmot("test") + manifest.uvy_track()
    for s in seqs:
        if (manifest.results_dir / "tracks" / MAIN_SET / "bytetrack" / f"{s.info.key}.txt").is_file():
            return s
    raise FileNotFoundError("no saved ByteTrack tracks for an available sequence")


def _frame(seq, number: int) -> np.ndarray:
    import cv2

    for f, img in seq.frames():
        if f == number:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise ValueError(f"frame {number} not found")


def plot_player_trajectories(manifest: Manifest, out: Path, window: int = 75) -> None:
    plt = _plt()
    from ..visualization import id_color

    seq = _overlay_sequence(manifest)
    tr = _proposed_tracks(manifest, seq)
    end = min(seq.info.num_frames, 150)
    start = max(1, end - window + 1)
    img = _frame(seq, end)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.imshow(img)
    shown = 0
    for tid in np.unique(tr[:, 1]):
        t = tr[(tr[:, 1] == tid) & (tr[:, 0] >= start) & (tr[:, 0] <= end)]
        if len(t) < 5:
            continue
        b, g, r = id_color(int(tid))
        color = (r / 255, g / 255, b / 255)
        foot_x, foot_y = t[:, 2] + t[:, 4] / 2, t[:, 3] + t[:, 5]
        ax.plot(foot_x, foot_y, color="black", linewidth=3.2, alpha=0.55, solid_capstyle="round")
        ax.plot(foot_x, foot_y, color=color, linewidth=1.8, solid_capstyle="round")
        if t[-1, 0] == end:
            x, y, w, h = t[-1, 2:6]
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.4))
            ax.text(x, y - 4, f"#{int(tid)}", color="black", fontsize=7, fontweight="bold",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1.0))
        shown += 1
    ax.set_xlim(0, seq.info.width)
    ax.set_ylim(seq.info.height, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(False)
    ax.set_title(f"Player trajectories (ByteTrack, proposed): {seq.info.dataset} {seq.info.name}, frames {start}-{end}\n"
                 f"{shown} tracks, foot-point trails in image coordinates (camera motion not compensated)")
    _save(fig, out)


def plot_player_heatmap(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    import cv2
    from matplotlib.colors import LinearSegmentedColormap

    seq = _overlay_sequence(manifest)
    tr = _proposed_tracks(manifest, seq)
    img = _frame(seq, 1)
    w, h = seq.info.width, seq.info.height
    cell = 8
    heat = np.zeros((h // cell + 1, w // cell + 1), dtype=np.float32)
    fx = np.clip((tr[:, 2] + tr[:, 4] / 2) / cell, 0, heat.shape[1] - 1).astype(int)
    fy = np.clip((tr[:, 3] + tr[:, 5]) / cell, 0, heat.shape[0] - 1).astype(int)
    np.add.at(heat, (fy, fx), 1.0)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=3.0)
    heat = heat / seq.info.num_frames * 1000.0  # player-detections per 1000 frames per cell (smoothed)
    cmap = LinearSegmentedColormap.from_list("one_hue", ["#fff3e8", "#f7b27f", "#eb6834", "#9c2f0a", "#4a1200"])
    alpha = np.clip(heat / (heat.max() + 1e-9), 0, 1) ** 0.5 * 0.85

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    ax.imshow(gray, alpha=0.75)
    im = ax.imshow(heat, cmap=cmap, alpha=alpha, extent=(0, heat.shape[1] * cell, heat.shape[0] * cell, 0),
                   interpolation="bilinear")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=im.norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Foot-point density (tracked positions per 1000 frames, per 8x8 px cell)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(False)
    ax.set_title(f"Player-position heatmap (ByteTrack, proposed): {seq.info.dataset} {seq.info.name}, "
                 f"{seq.info.num_frames} frames\nimage coordinates over the first frame (grey)")
    _save(fig, out)


# --------------------------------------------------------------------------- #
# 8-9: ball
# --------------------------------------------------------------------------- #
def _ball_trace_sequence(manifest: Manifest):
    """UVY sequence whose Kalman track contains the most predicted frames (falls back to any ball sequence)."""
    best, best_n = None, -1
    for seq in manifest.uvy_track() + manifest.infactory():
        p = manifest.results_dir / "metrics" / "ball_traces" / f"{seq.info.key}.csv"
        if not p.is_file():
            continue
        rows = read_csv(p)
        n = sum(1 for r in rows if r["trk_status"] == "predicted") + (10000 if seq.info.dataset == "UVY-Track" else 0)
        if n > best_n:
            best, best_n = (seq, rows), n
    if best is None:
        raise FileNotFoundError("no ball traces found")
    return best


def plot_ball_trajectory(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    seq, rows = _ball_trace_sequence(manifest)
    frames = np.array([r["frame"] for r in rows])
    det = [r for r in rows if r["trk_status"] == "detected"]
    pred = [r for r in rows if r["trk_status"] == "predicted"]
    lost = [r for r in rows if r["trk_status"] == "lost"]
    gt = [r for r in rows if r["ball_visible"]]
    mid = det[len(det) // 2]["frame"] if det else 1
    img = _frame(seq, mid)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10.5), gridspec_kw={"height_ratios": [1.55, 1]})
    ax1.imshow(img, alpha=0.8)
    ax1.plot([r["gt_cx"] for r in gt], [r["gt_cy"] for r in gt], color="white", linewidth=3.0, alpha=0.9)
    ax1.plot([r["gt_cx"] for r in gt], [r["gt_cy"] for r in gt], color=BALL_COLORS["gt"], linewidth=1.2,
             label="Ground truth")
    ax1.scatter([r["trk_cx"] for r in det], [r["trk_cy"] for r in det], s=16, color=BALL_COLORS["detected"],
                edgecolors="white", linewidths=0.4, label=f"Detected ({len(det)} frames)", zorder=3)
    ax1.scatter([r["trk_cx"] for r in pred], [r["trk_cy"] for r in pred], s=46, marker="^", facecolors="none",
                edgecolors=BALL_COLORS["predicted"], linewidths=1.6,
                label=f"Predicted by Kalman filter ({len(pred)} frames)", zorder=4)
    ax1.set_xlim(0, seq.info.width)
    ax1.set_ylim(seq.info.height, 0)
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.grid(False)
    ax1.legend(loc="upper right", frameon=True, facecolor=SURFACE, framealpha=0.9)
    ax1.set_title(f"Ball trajectory: {seq.info.dataset} {seq.info.name} ({len(rows)} frames), drawn over frame {mid}")

    ax2.plot([r["frame"] for r in gt], [r["gt_cx"] for r in gt], color=BALL_COLORS["gt"], linewidth=1.2,
             label="Ground truth")
    ax2.scatter([r["frame"] for r in det], [r["trk_cx"] for r in det], s=12, color=BALL_COLORS["detected"],
                label="Detected", zorder=3)
    ax2.scatter([r["frame"] for r in pred], [r["trk_cx"] for r in pred], s=40, marker="^", facecolors="none",
                edgecolors=BALL_COLORS["predicted"], linewidths=1.5, label="Predicted", zorder=4)
    y0 = -0.035 * seq.info.width
    if lost:
        ax2.scatter([r["frame"] for r in lost], [y0] * len(lost), s=28, marker="|", color="#e34948",
                    label=f"Lost: no output ({len(lost)} frames)")
    ax2.set_xlim(frames.min() - 1, frames.max() + 1)
    ax2.set_ylim(2 * y0, seq.info.width)
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Ball centre x (pixels)")
    ax2.set_title("Ball x-position over time: detected, predicted and lost frames")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    _save(fig, out)


def plot_ball_status(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    rows = [r for r in read_csv(manifest.results_dir / "metrics" / "ball_per_sequence.csv") if r["method"] == "kalman"]
    rows = [r for r in rows if r["frames_labelled"] >= 30]
    labels = [f"{r['sequence']} ({r['dataset']}, {r['frames_labelled']} fr.)" for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 1.9))
    left = np.zeros(len(rows))
    for key, name in (("frames_detected", "detected"), ("frames_predicted", "predicted"), ("frames_lost", "lost")):
        vals = np.array([100.0 * r[key] / r["frames_labelled"] for r in rows])
        ax.barh(labels, vals, left=left, color=BALL_COLORS[name], edgecolor=SURFACE, linewidth=1.5,
                label={"detected": "Detected", "predicted": "Predicted (Kalman)", "lost": "Lost (no output)"}[name])
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 7:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if name != "lost" else INK)
        left += vals
    vis = [100.0 * r["frames_ball_visible"] / r["frames_labelled"] for r in rows]
    ax.scatter(vis, range(len(rows)), marker="D", s=30, color=INK, zorder=5, label="Share of frames with a visible ball")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of labelled frames (%)")
    ax.set_ylabel("Sequence")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title("Ball track status per sequence (Kalman ball tracker): detected, predicted, lost")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16 if len(rows) > 6 else -0.3), ncol=4)
    _save(fig, out)


def plot_ball_pr_curve(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    curves = read_json(manifest.results_dir / "metrics" / "ball_pr_curves.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, c in curves.items():
        ax.plot(c["recall"], c["precision"], linewidth=2, color=DATASET_COLORS.get(name, INK2),
                label=f"{name}: AP@0.5 = {c['ap50']:.1f}% ({c['num_gt']} balls)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.grid(axis="x", visible=True)
    ax.set_title("Ball detector precision-recall (YOLO11n 'sports ball', IoU >= 0.5)")
    ax.legend(loc="upper right")
    _save(fig, out)


# --------------------------------------------------------------------------- #
# extras: tuning sensitivity, model comparison, architecture
# --------------------------------------------------------------------------- #
def plot_tuning(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    rows = read_csv(manifest.results_dir / "metrics" / "player_tuning_grid.csv")
    sel = read_json(manifest.results_dir / "metrics" / "player_selected_params.json")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    rng = np.random.default_rng(0)
    for i, a in enumerate(ALGORITHMS):
        vals = np.array([r["HOTA"] for r in rows if r["tracker"] == a])
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=18, color=TRACKER_COLORS[a], alpha=0.55,
                   linewidths=0)
        best = vals.max()
        ax.scatter([i], [best], s=90, marker="*", color=INK, zorder=5, label="Selected (best HOTA)" if i == 0 else None)
        ax.text(i + 0.22, best, f"{best:.1f}", va="center", fontsize=9)
        dflt = next((r["HOTA"] for r in rows if r["tracker"] == a
                     and all(r[k] == v for k, v in sel["default"].items())), None)
        if dflt is not None:
            ax.scatter([i], [dflt], s=60, marker="o", facecolors="none", edgecolors=INK, linewidths=1.4, zorder=5,
                       label="Project default parameters" if i == 0 else None)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([SHORT_LABELS[a] for a in ALGORITHMS])
    ax.set_xlim(-0.5, len(ALGORITHMS) - 0.3)
    ax.set_xlabel("Tracker")
    ax.set_ylabel("HOTA on the dev sequence (%)")
    ax.set_title("Hyper-parameter sensitivity on the dev sequence\n(48 grid points per tracker; "
                 + ", ".join(sel["dev_sequences"]) + ")")
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", length=0)
    _save(fig, out)


def plot_model_comparison(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    rows = read_csv(manifest.results_dir / "metrics" / "model_comparison.csv")
    models = list(dict.fromkeys(r["model"] for r in rows))
    colors = dict(zip(models, ["#6250d6", "#1baf7a", "#eda100"]))
    panels = [("player", ["precision", "recall", "f1", "ap50"], "Players (SportsMOT subset)"),
              ("ball", ["precision", "recall", "f1", "ap50"], "Ball (InFactory subset)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for ax, (target, keys, title) in zip(axes, panels):
        series = [(m, colors[m], [next(r[k] for r in rows if r["model"] == m and r["target"] == target) for k in keys])
                  for m in models]
        _grouped_bars(ax, ["Precision", "Recall", "F1", "AP@0.5"], series, "Score (%), higher is better",
                      ylim=(0, 112))
        ax.set_title(title)
    ms = [next(r["detector_ms_per_frame"] for r in rows if r["model"] == m and r["target"] == "player") for m in models]
    bars = axes[2].bar(models, ms, 0.45, color=[colors[m] for m in models], linewidth=0)
    for b, v in zip(bars, ms):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    axes[2].set_ylabel("Detector latency (ms per frame), lower is better")
    axes[2].set_title("Inference time at 1280 px (CPU)")
    axes[2].set_ylim(0, max(ms) * 1.18)
    axes[2].tick_params(axis="x", length=0)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=len(models))
    fig.suptitle("Detector comparison: YOLOv8n vs YOLO11n (same sequences, same thresholds)", fontsize=12,
                 fontweight="bold", y=1.02)
    _save(fig, out)


def plot_architecture(manifest: Manifest, out: Path) -> None:
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 54)
    ax.axis("off")

    def node(x, y, w, h, title, sub, color="#ffffff", edge=GRID, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", facecolor=color,
                                    edgecolor=edge, linewidth=1.6 if bold else 1.1))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, sub, ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.35)

    def arrow(a, b, label=None, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=INK2,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 1.3, label, ha="center", fontsize=7.2, color=INK2)

    node(1, 21, 17, 13, "Video input", "MP4 / AVI / frames\nvalidation, FPS,\ntimestamps")
    node(23, 21, 21, 13, "YOLO11n detector", "one shared pass, 1280 px\nclasses: person,\nsports ball", "#eef4fc",
         "#2a78d6", True)
    node(50, 36, 21, 13, "Player filter", "height, aspect ratio,\noptional pitch mask\n(conf >= 0.15)")
    node(50, 5, 21, 13, "Ball filter", "size, squareness,\noptional pitch mask\n(conf >= 0.10)")
    node(77, 36, 21, 13, "ByteTrack", "two-stage association\n(high + low confidence)\npersistent player IDs", "#e8f7f1",
         "#1baf7a", True)
    node(77, 5, 21, 13, "Kalman ball tracker", "constant velocity, gating,\npredicts through gaps,\nre-acquisition", "#e8f7f1",
         "#1baf7a", True)
    node(104, 36, 20, 13, "Annotated video", "boxes, IDs, trails,\ndetected vs predicted\nball styling")
    node(104, 20.5, 20, 13, "CSV + JSON", "per-frame tracks,\nrun summary")
    node(104, 5, 20, 13, "Evaluation + SQLite", "HOTA / MOTA / IDF1,\nball P / R / F1,\nplots, database")

    arrow((18.4, 27.5), (22.6, 27.5), "frames")
    arrow((44.4, 30), (49.6, 42.5), "person", 0.0)
    arrow((44.4, 25), (49.6, 11.5), "sports ball", 0.0)
    arrow((71.4, 42.5), (76.6, 42.5))
    arrow((71.4, 11.5), (76.6, 11.5))
    arrow((98.4, 42.5), (103.6, 42.5))
    arrow((98.4, 40), (103.6, 29))
    arrow((98.4, 13), (103.6, 25))
    arrow((98.4, 11.5), (103.6, 11.5))
    arrow((98.4, 15), (103.6, 39), rad=-0.12)
    ax.text(87.5, 28, "Proposed hybrid tracking:\nmulti-object association for players,\n"
                      "single-target motion model for the ball", ha="center", va="center", fontsize=8,
            color="#0d7a54", style="italic")
    ax.text(87.5, 51.5, "Baselines swap this block for SORT or OC-SORT", ha="center", fontsize=7.6, color=INK2)
    ax.set_title("Proposed system: YOLO11n + ByteTrack (players) + Kalman filter (ball)", fontsize=12,
                 fontweight="bold")
    _save(fig, out)


PLOTS: list[tuple[str, Callable[[Manifest, Path], None], str]] = [
    ("00_architecture.png", plot_architecture, "System architecture diagram"),
    ("01_hota_mota_idf1.png", plot_hota_mota_idf1, "HOTA, MOTA and IDF1 for SORT, OC-SORT and ByteTrack"),
    ("02_deta_assa.png", plot_deta_assa, "DetA and AssA comparison"),
    ("03_detection_prf.png", plot_detection_prf, "Player / ball detection precision, recall, F1"),
    ("04_runtime.png", plot_runtime, "Tracker latency and end-to-end FPS"),
    ("05_idsw_frag.png", plot_idsw_frag, "ID switches and track fragmentation"),
    ("06_player_trajectories.png", plot_player_trajectories, "Player trajectories over a video frame"),
    ("07_player_heatmap.png", plot_player_heatmap, "Player-position heatmap"),
    ("08_ball_trajectory.png", plot_ball_trajectory, "Ball trajectory: detected, predicted, lost"),
    ("09_ball_status.png", plot_ball_status, "Ball track status per sequence"),
    ("10_ball_pr_curve.png", plot_ball_pr_curve, "Ball detector precision-recall curve"),
    ("11_per_sequence_hota.png", plot_per_sequence_hota, "HOTA per test sequence"),
    ("12_tuning_sensitivity.png", plot_tuning, "Hyper-parameter sensitivity on the dev sequence"),
    ("13_model_comparison.png", plot_model_comparison, "YOLOv8n vs YOLO11n"),
]


def make_all_plots(manifest: Manifest) -> None:
    out_dir = manifest.results_dir / "plots"
    index = []
    for filename, fn, description in PLOTS:
        try:
            fn(manifest, out_dir / filename)
            index.append({"file": filename, "description": description, "status": "ok"})
            log.info("plot %s", filename)
        except (FileNotFoundError, StopIteration) as exc:
            index.append({"file": filename, "description": description, "status": f"skipped: {exc}"})
            log.warning("plot %s skipped (inputs missing: %s)", filename, exc)
    write_json(out_dir / "index.json", index)
