"""Trajectory figures for one application run (uses the run's tracks CSV and its input video).

    python scripts/visualize_run.py --csv results/demo/uvy_soccer_V04_tracks.csv \
        --video data/raw/uvy_track/soccer_V04/vid__fkPwtg7ozU.mp4 --out-dir results/demo --prefix uvy_soccer_V04

Writes <prefix>_player_trajectories.png, <prefix>_player_heatmap.png,
<prefix>_ball_trajectory.png and <prefix>_screenshot.png.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from soccer_tracker.experiments.plots import BALL_COLORS, SURFACE, _plt
from soccer_tracker.visualization import id_color


def read_frame(video: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    try:
        img = None
        for _ in range(index + 1):
            ok, frame = cap.read()
            if not ok:
                break
            img = frame
    finally:
        cap.release()
    if img is None:
        raise SystemExit(f"Could not read frame {index} from {video}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="tracks CSV written by the application")
    ap.add_argument("--video", required=True, help="the input video of that run")
    ap.add_argument("--annotated", default=None, help="annotated output video (for the screenshot)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="run")
    ap.add_argument("--frame", type=int, default=None, help="background frame index (default: middle of the run)")
    args = ap.parse_args()

    players: dict[int, list] = defaultdict(list)
    ball: list[dict] = []
    frames_seen: set[int] = set()
    with open(args.csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            frames_seen.add(f)
            if r["object_type"] == "player":
                players[int(r["track_id"])].append((f, float(r["center_x"]), float(r["y2"])))
            else:
                ball.append({"frame": f, "x": float(r["center_x"]), "y": float(r["center_y"]), "status": r["status"]})
    if not frames_seen:
        print("The CSV has no rows.", file=sys.stderr)
        return 2
    # Frames after the last CSV row had no tracked object at all; count them as "lost" too.
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    last = max(max(frames_seen), total - 1)
    bg_index = args.frame if args.frame is not None else last // 2
    bg = read_frame(args.video, bg_index)
    h, w = bg.shape[:2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _plt()

    # 1. player trajectories
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.imshow(bg, alpha=0.85)
    shown = 0
    for tid, pts in sorted(players.items()):
        if len(pts) < 10:
            continue
        pts.sort()
        b, g, r = id_color(tid)
        color = (r / 255, g / 255, b / 255)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(xs, ys, color="black", linewidth=3.0, alpha=0.5, solid_capstyle="round")
        ax.plot(xs, ys, color=color, linewidth=1.7, solid_capstyle="round")
        ax.text(xs[-1], ys[-1], f"#{tid}", fontsize=7, fontweight="bold", color="black",
                bbox=dict(facecolor=color, edgecolor="none", pad=1.0))
        shown += 1
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(False)
    ax.set_title(f"Player trajectories, frames 0-{last}: {shown} tracks of at least 10 frames\n"
                 f"foot-point trails in image coordinates over frame {bg_index}")
    fig.savefig(out_dir / f"{args.prefix}_player_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    # 2. player heatmap
    from matplotlib.colors import LinearSegmentedColormap

    cell = 8
    heat = np.zeros((h // cell + 1, w // cell + 1), np.float32)
    for pts in players.values():
        for _, x, y in pts:
            heat[min(int(y / cell), heat.shape[0] - 1), min(max(int(x / cell), 0), heat.shape[1] - 1)] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=3.0)
    cmap = LinearSegmentedColormap.from_list("one_hue", ["#fff3e8", "#f7b27f", "#eb6834", "#9c2f0a", "#4a1200"])
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.imshow(cv2.cvtColor(cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB), alpha=0.75)
    alpha = np.clip(heat / (heat.max() + 1e-9), 0, 1) ** 0.5 * 0.85
    im = ax.imshow(heat, cmap=cmap, alpha=alpha, extent=(0, heat.shape[1] * cell, heat.shape[0] * cell, 0),
                   interpolation="bilinear")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=im.norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Foot-point density (tracked positions per 8x8 px cell, smoothed)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(False)
    ax.set_title(f"Player-position heatmap, frames 0-{last} (image coordinates)")
    fig.savefig(out_dir / f"{args.prefix}_player_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # 3. ball trajectory
    det = [b for b in ball if b["status"] == "detected"]
    pred = [b for b in ball if b["status"] == "predicted"]
    have = {b["frame"] for b in ball}
    lost = [f for f in range(last + 1) if f not in have]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10.2), gridspec_kw={"height_ratios": [1.55, 1]})
    ax1.imshow(bg, alpha=0.8)
    ax1.scatter([b["x"] for b in det], [b["y"] for b in det], s=16, color=BALL_COLORS["detected"], edgecolors="white",
                linewidths=0.4, label=f"Detected ({len(det)} frames)", zorder=3)
    ax1.scatter([b["x"] for b in pred], [b["y"] for b in pred], s=46, marker="^", facecolors="none",
                edgecolors=BALL_COLORS["predicted"], linewidths=1.6, label=f"Predicted by Kalman filter ({len(pred)} frames)",
                zorder=4)
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.grid(False)
    ax1.legend(loc="upper right", frameon=True, facecolor=SURFACE, framealpha=0.9)
    ax1.set_title(f"Ball trajectory over frame {bg_index} (no ground truth: this is the system's own output)")
    ax2.scatter([b["frame"] for b in det], [b["x"] for b in det], s=12, color=BALL_COLORS["detected"], label="Detected")
    ax2.scatter([b["frame"] for b in pred], [b["x"] for b in pred], s=40, marker="^", facecolors="none",
                edgecolors=BALL_COLORS["predicted"], linewidths=1.5, label="Predicted")
    y0 = -0.035 * w
    if lost:
        ax2.scatter(lost, [y0] * len(lost), s=28, marker="|", color="#e34948", label=f"Lost: no output ({len(lost)} frames)")
    ax2.set_xlim(-1, last + 1)
    ax2.set_ylim(2 * y0, w)
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Ball centre x (pixels)")
    ax2.set_title("Ball x-position over time: detected, predicted and lost frames")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.prefix}_ball_trajectory.png", bbox_inches="tight")
    plt.close(fig)

    # 4. screenshot of the annotated video (a frame where the ball is tracked, if any)
    if args.annotated:
        target = det[len(det) // 2]["frame"] if det else bg_index
        shot = read_frame(args.annotated, target)
        cv2.imwrite(str(out_dir / f"{args.prefix}_screenshot.png"), cv2.cvtColor(shot, cv2.COLOR_RGB2BGR))
    print(f"Figures written to {out_dir} with prefix '{args.prefix}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
