"""End-to-end orchestration: read -> detect -> filter -> track -> draw -> write.

The pipeline only talks to components through small interfaces
(``Detector.detect``, ``PlayerTracker.update``, ``BallTracker.update``), so
any stage can be replaced independently - which is also how the smoke test
runs without downloading model weights.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .analytics import RunStats, TrackCsvWriter, write_summary
from .config import AppConfig, config_to_dict
from .detection.base import Detector
from .detection.filters import PitchMasker, filter_balls, filter_players
from .errors import ProcessingError, SoccerTrackerError, VideoWriteError
from .tracking.ball_tracker import BallTracker
from .tracking.player_tracker import PlayerTracker
from .video_io import VideoReader, VideoWriter
from .visualization import Visualizer

log = logging.getLogger(__name__)

DetectorFactory = Callable[[AppConfig], tuple[Detector, Optional[Detector]]]


def build_yolo_detectors(cfg: AppConfig) -> tuple[Detector, Optional[Detector]]:
    """Default factory: Ultralytics YOLO for players and (optionally) the ball."""
    from .detection.yolo import ModelCache, YoloDetector, build_shared_detectors, can_share_inference

    cache = ModelCache()
    if not cfg.ball.enabled:
        return YoloDetector(cfg.players.detector, cache), None
    if cfg.ball.share_player_inference and can_share_inference(cfg.players.detector, cfg.ball.detector):
        log.info("Players and ball share one YOLO pass per frame (ball.share_player_inference)")
        return build_shared_detectors(cfg.players.detector, cfg.ball.detector, cache)
    return YoloDetector(cfg.players.detector, cache), YoloDetector(cfg.ball.detector, cache)


class Pipeline:
    def __init__(self, cfg: AppConfig, detector_factory: DetectorFactory = build_yolo_detectors):
        self.cfg = cfg
        self.detector_factory = detector_factory

    def run(
        self,
        input_path: str | Path,
        output_video: Optional[str | Path] = None,
        output_csv: Optional[str | Path] = None,
        output_summary: Optional[str | Path] = None,
    ) -> dict:
        cfg = self.cfg
        input_path = Path(input_path)
        out_video = Path(output_video or cfg.io.output_video)
        out_csv = Path(output_csv or cfg.io.output_csv)
        summary_path = output_summary if output_summary is not None else cfg.io.output_summary
        summary_path = Path(summary_path) if summary_path else None

        for p in (out_video, out_csv):
            if input_path.exists() and p.exists() and p.resolve() == input_path.resolve():
                raise VideoWriteError(f"Output path '{p}' is the same as the input video; refusing to overwrite it.")
        if out_video.exists() and not cfg.io.overwrite:
            raise VideoWriteError(f"Output '{out_video}' exists and io.overwrite is false.")

        stats = RunStats()
        with VideoReader(input_path, cfg.io.fallback_fps, cfg.io.max_frames) as reader:
            info = reader.info
            assert info is not None
            # Load models only after the video is known to be readable, and
            # before any output file is created.
            player_detector, ball_detector = self.detector_factory(cfg)

            player_tracker = PlayerTracker(cfg.players.tracker, frame_rate=info.fps)
            ball_tracker = BallTracker(cfg.ball.tracker) if ball_detector is not None else None
            visualizer = Visualizer(cfg.visualization)
            needs_mask = cfg.players.filter.use_pitch_mask or (ball_detector is not None and cfg.ball.filter.use_pitch_mask)
            masker = PitchMasker(cfg.pitch_mask) if needs_mask else None

            t0 = time.perf_counter()
            frame_idx = -1
            try:
                with VideoWriter(out_video, info.fps, (info.width, info.height), cfg.io.fourcc, cfg.io.overwrite) as writer, \
                        TrackCsvWriter(out_csv) as csv_writer:
                    for frame_idx, ts, frame in reader.frames():
                        mask = masker.compute(frame) if masker else None

                        raw_p = player_detector.detect(frame)
                        kept_p = filter_players(raw_p, frame.shape, cfg.players.filter, mask)
                        players = player_tracker.update(kept_p)
                        stats.player_detections_raw += len(raw_p)
                        stats.player_detections_kept += len(kept_p)

                        ball = None
                        traj = ()
                        if ball_detector is not None and ball_tracker is not None:
                            raw_b = ball_detector.detect(frame)
                            kept_b = filter_balls(raw_b, cfg.ball.filter, mask)
                            ball = ball_tracker.update(frame_idx, kept_b, frame.shape)
                            traj = ball_tracker.trajectory if ball is not None else ()
                            stats.ball_candidates_raw += len(raw_b)
                            stats.ball_candidates_kept += len(kept_b)

                        writer.write(visualizer.draw(frame, frame_idx, ts, players, ball, traj))
                        csv_writer.write_frame(frame_idx, ts, players + ([ball] if ball is not None else []))
                        stats.record(players, ball)

                        if (frame_idx + 1) % cfg.io.log_every == 0:
                            el = time.perf_counter() - t0
                            limits = [n for n in (info.frame_count, cfg.io.max_frames) if n]
                            total = min(limits) if limits else None
                            log.info(
                                "frame %d/%s  %.1f FPS  players=%d  ball=%s",
                                frame_idx + 1, total or "?", (frame_idx + 1) / el, len(players),
                                ball.status if ball else "lost",
                            )
            except SoccerTrackerError:
                raise
            except KeyboardInterrupt:
                log.warning("Interrupted at frame %d; partial outputs were written and closed.", frame_idx)
                raise
            except Exception as exc:
                raise ProcessingError(
                    f"Processing failed at frame {frame_idx} ({type(exc).__name__}: {exc}). "
                    f"Partial outputs may exist at '{out_video}' and '{out_csv}'."
                ) from exc

            stats.elapsed_s = time.perf_counter() - t0
            if ball_tracker is not None:
                stats.ball_tracks_started = ball_tracker.tracks_started

        if stats.frames == 0:
            raise ProcessingError("No frames were processed.")

        summary = stats.to_dict(
            {
                "input": str(input_path),
                "video": {"width": info.width, "height": info.height, "fps": info.fps,
                          "fps_was_fallback": info.fps_was_fallback, "reported_frame_count": info.frame_count},
                "outputs": {"video": str(out_video), "csv": str(out_csv),
                            "summary": str(summary_path) if summary_path else None},
                "config": config_to_dict(cfg),
            }
        )
        if summary_path:
            write_summary(summary_path, summary)
        log.info(
            "Done: %d frames in %.1fs (%.1f FPS). %d player IDs, ball coverage %.0f%%. Video: %s  CSV: %s",
            stats.frames, stats.elapsed_s, summary["processing_fps"] or 0, summary["players"]["unique_track_ids"],
            100 * summary["ball"]["coverage"], out_video, out_csv,
        )
        return summary
