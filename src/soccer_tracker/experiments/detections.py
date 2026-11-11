"""Run the detector once per sequence and cache the result.

Why cache? The baseline comparison must give every tracker *identical*
detections, so the detector is run exactly once per frame and the three
trackers replay the stored boxes. It also makes the comparison cheap: on a CPU
the detector takes ~0.6 s per frame while a tracker takes about a millisecond.

The cache stores the detector output *before* the geometric player/ball
filters, at the same confidence thresholds the application uses, so replaying
it reproduces the application's detection stage exactly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..datasets import Sequence
from ..detection.base import Detections

log = logging.getLogger(__name__)

PERSON = "person"
BALL = "sports ball"


@dataclass
class DetectionCache:
    meta: dict
    infer_s: np.ndarray  # (N,) detector seconds per frame, index 0 = frame 1
    p_frame: np.ndarray  # (P,) frame number of each player detection (sorted)
    p_xyxy: np.ndarray
    p_conf: np.ndarray
    b_frame: np.ndarray  # (B,) frame number of each ball candidate (sorted)
    b_xyxy: np.ndarray
    b_conf: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.infer_s)

    def _slice(self, frames: np.ndarray, frame: int) -> slice:
        return slice(int(np.searchsorted(frames, frame, "left")), int(np.searchsorted(frames, frame, "right")))

    def players(self, frame: int) -> Detections:
        s = self._slice(self.p_frame, frame)
        n = s.stop - s.start
        return Detections(self.p_xyxy[s], self.p_conf[s], np.array([PERSON] * n, dtype=object))

    def balls(self, frame: int) -> Detections:
        s = self._slice(self.b_frame, frame)
        n = s.stop - s.start
        return Detections(self.b_xyxy[s], self.b_conf[s], np.array([BALL] * n, dtype=object))

    def detector_time(self) -> dict:
        """Timing summary. The first frame includes model warm-up and is excluded."""
        t = self.infer_s[1:] if len(self.infer_s) > 1 else self.infer_s
        return {
            "mean_ms": float(np.mean(t) * 1000),
            "median_ms": float(np.median(t) * 1000),
            "p95_ms": float(np.percentile(t, 95) * 1000),
            "total_s": float(np.sum(self.infer_s)),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, meta=json.dumps(self.meta), infer_s=self.infer_s,
            p_frame=self.p_frame, p_xyxy=self.p_xyxy, p_conf=self.p_conf,
            b_frame=self.b_frame, b_xyxy=self.b_xyxy, b_conf=self.b_conf,
        )

    @classmethod
    def load(cls, path: Path) -> "DetectionCache":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                meta=json.loads(str(z["meta"])), infer_s=z["infer_s"],
                p_frame=z["p_frame"], p_xyxy=z["p_xyxy"], p_conf=z["p_conf"],
                b_frame=z["b_frame"], b_xyxy=z["b_xyxy"], b_conf=z["b_conf"],
            )


def cache_path(results_dir: Path, seq: Sequence, tag: str) -> Path:
    return results_dir / "cache" / "detections" / f"{seq.info.key}__{tag}.npz"


def compute_detections(seq: Sequence, cfg: AppConfig, detectors=None, log_every: int = 100) -> DetectionCache:
    """Run the application's detection stage over a sequence."""
    from ..pipeline import build_yolo_detectors

    player_det, ball_det = detectors or build_yolo_detectors(cfg)
    infer, pf, px, pc, bf, bx, bc = [], [], [], [], [], [], []
    t_start = time.perf_counter()
    for f, img in seq.frames():
        t0 = time.perf_counter()
        p = player_det.detect(img)
        b = ball_det.detect(img) if ball_det is not None else Detections.empty()
        infer.append(time.perf_counter() - t0)
        if len(p):
            pf.append(np.full(len(p), f)); px.append(p.xyxy); pc.append(p.confidence)
        if len(b):
            bf.append(np.full(len(b), f)); bx.append(b.xyxy); bc.append(b.confidence)
        if f % log_every == 0:
            el = time.perf_counter() - t_start
            log.info("  %s frame %d/%d  (%.2f FPS)", seq.info.name, f, seq.info.num_frames, f / el)

    def cat(parts, shape, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype=dtype)

    d = cfg.players.detector
    meta = {
        "dataset": seq.info.dataset, "sequence": seq.info.name, "frames": seq.info.num_frames,
        "weights": Path(d.weights).name, "imgsz": d.imgsz, "device": d.device,
        "player_classes": d.classes, "player_confidence": d.confidence, "player_iou": d.iou,
        "ball_classes": cfg.ball.detector.classes, "ball_confidence": cfg.ball.detector.confidence,
        "ball_tiling": cfg.ball.detector.tiling.enabled,
        "shared_inference": cfg.ball.share_player_inference,
    }
    return DetectionCache(
        meta=meta, infer_s=np.asarray(infer, dtype=np.float64),
        p_frame=cat(pf, (0,), np.int64), p_xyxy=cat(px, (0, 4), np.float32), p_conf=cat(pc, (0,), np.float32),
        b_frame=cat(bf, (0,), np.int64), b_xyxy=cat(bx, (0, 4), np.float32), b_conf=cat(bc, (0,), np.float32),
    )


def get_detections(seq: Sequence, cfg: AppConfig, results_dir: Path, tag: str, detectors=None,
                   compute: bool = True) -> DetectionCache | None:
    """Load cached detections, computing them first if needed."""
    path = cache_path(results_dir, seq, tag)
    if path.is_file():
        cache = DetectionCache.load(path)
        if cache.num_frames == seq.info.num_frames:
            return cache
        log.warning("Cache %s has %d frames, expected %d; recomputing", path.name, cache.num_frames,
                    seq.info.num_frames)
    if not compute:
        return None
    log.info("Detecting %s / %s (%d frames) with %s", seq.info.dataset, seq.info.name, seq.info.num_frames, tag)
    cache = compute_detections(seq, cfg, detectors)
    cache.save(path)
    return cache
