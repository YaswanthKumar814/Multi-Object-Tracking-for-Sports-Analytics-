"""Reproduce every experiment in the report.

    python scripts/run_experiments.py all          # everything below, in order
    python scripts/run_experiments.py detect       # run YOLO once per sequence and cache detections
    python scripts/run_experiments.py runtime      # time every pipeline stage (run on an idle machine)
    python scripts/run_experiments.py players      # SORT vs OC-SORT vs ByteTrack on cached detections
    python scripts/run_experiments.py ball         # detection-only vs Kalman ball tracking
    python scripts/run_experiments.py models       # optional: YOLOv8n vs YOLO11n
    python scripts/run_experiments.py plots        # figures from the stored results
    python scripts/run_experiments.py db           # (re)build results/sports_analytics.db

Which sequences are used is defined in configs/experiments.yaml. Stages read the
outputs of earlier stages from results/, so they can be re-run independently.
The slow stage is ``detect`` (about 0.6 s per frame on a CPU); it is resumable.
"""

from __future__ import annotations

import argparse
import logging
import sys

from soccer_tracker.experiments import get_detections, load_manifest

log = logging.getLogger("experiments")

STAGES = ["detect", "runtime", "players", "ball", "models", "plots", "db"]


def stage_detect(manifest) -> None:
    from soccer_tracker.pipeline import build_yolo_detectors

    cfg = manifest.app_config()
    tag = manifest.detector_tag()
    detectors = build_yolo_detectors(cfg)
    sequences = manifest.all_sequences()
    total = sum(s.info.num_frames for s in sequences)
    log.info("Detection stage: %d sequences, %d frames, detector %s", len(sequences), total, tag)
    for seq in sequences:
        cache = get_detections(seq, cfg, manifest.results_dir, tag, detectors)
        t = cache.detector_time()
        log.info("%-10s %-22s %5d frames  detector %.0f ms/frame (median)", seq.info.dataset, seq.info.name,
                 cache.num_frames, t["median_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--manifest", default="configs/experiments.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    manifest = load_manifest(args.manifest)
    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "detect":
            stage_detect(manifest)
        elif stage == "runtime":
            from soccer_tracker.experiments.runtime import run_runtime_benchmark
            run_runtime_benchmark(manifest)
        elif stage == "players":
            from soccer_tracker.experiments.players import run_player_experiments
            run_player_experiments(manifest)
        elif stage == "ball":
            from soccer_tracker.experiments.ball import run_ball_experiments
            run_ball_experiments(manifest)
        elif stage == "models":
            from soccer_tracker.experiments.models import run_model_comparison
            run_model_comparison(manifest)
        elif stage == "plots":
            from soccer_tracker.experiments.plots import make_all_plots
            make_all_plots(manifest)
        elif stage == "db":
            from soccer_tracker.experiments.database import build_database
            build_database(manifest)
    for note in manifest.skipped:
        log.warning("SKIPPED %s", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
