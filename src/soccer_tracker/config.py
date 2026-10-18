"""Typed configuration loaded from YAML.

Design notes
------------
* Every option has a default here, so a YAML file only needs to list what it
  changes. ``configs/default.yaml`` mirrors these defaults for discoverability.
* Unknown keys are rejected: a typo such as ``confidance`` fails loudly instead
  of being silently ignored.
* Relative paths are resolved against the current working directory at run
  time, never against a hard-coded location.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .errors import ConfigError


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class TilingConfig:
    """Sliced inference: run the detector on overlapping crops so small
    objects (the ball) are not shrunk away when the frame is resized."""

    enabled: bool = False
    tile_size: int = 640
    overlap: float = 0.2  # fraction of tile_size shared by neighbouring tiles
    include_full_frame: bool = True  # also run once on the whole frame


@dataclass
class DetectorConfig:
    weights: str = "weights/yolo11n.pt"
    # Class *names* as reported by the model (model.names). Names are used
    # instead of numeric ids so swapping in a custom model (e.g. one whose
    # ball class is "ball") is a config change, not a code change.
    classes: list[str] = field(default_factory=lambda: ["person"])
    confidence: float = 0.15
    iou: float = 0.5
    imgsz: int = 1280
    device: Optional[str] = None  # None = let Ultralytics choose (GPU if available)
    half: bool = False  # FP16; only honoured on CUDA
    allow_download: bool = True  # allow Ultralytics to fetch official weights
    tiling: TilingConfig = field(default_factory=TilingConfig)


@dataclass
class PlayerFilterConfig:
    min_height_px: float = 12.0
    max_height_frac: float = 0.6  # of frame height; rejects huge close-ups/crowd blobs
    min_aspect: float = 0.8  # height / width
    max_aspect: float = 6.0
    use_pitch_mask: bool = False


@dataclass
class ByteTrackConfig:
    """Parameters passed to ``trackers.ByteTrackTracker`` (Roboflow, Apache-2.0).

    ByteTrack associates high-confidence detections first, then tries to
    rescue unmatched tracks with low-confidence detections. That second stage
    only helps if the detector's own threshold (players.detector.confidence)
    is set *below* ``high_conf_det_threshold``.
    """

    track_activation_threshold: float = 0.5  # min confidence to start a new track
    high_conf_det_threshold: float = 0.5  # split between first/second association stage
    minimum_iou_threshold: float = 0.1  # min IoU for a detection-track match
    lost_track_buffer: int = 30  # frames (at 30 FPS) a lost track is kept for re-association
    minimum_consecutive_frames: int = 3  # frames before a new track gets an ID


@dataclass
class PlayersConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    filter: PlayerFilterConfig = field(default_factory=PlayerFilterConfig)
    tracker: ByteTrackConfig = field(default_factory=ByteTrackConfig)


@dataclass
class BallFilterConfig:
    min_size_px: float = 2.0
    max_size_px: float = 80.0
    max_aspect_ratio: float = 2.0  # a ball box should be roughly square
    use_pitch_mask: bool = False


@dataclass
class BallTrackerConfig:
    """Single-target Kalman tracker (constant-velocity model)."""

    gate_px: float = 80.0  # max distance between prediction and accepted detection
    gate_growth_px: float = 20.0  # gate widens by this much per missed frame
    max_missed_frames: int = 15  # frames to coast on prediction before dropping
    min_hits: int = 2  # detections needed before a track is reported
    reinit_confidence: float = 0.5  # a detection this strong outside the gate restarts the track
    reinit_after_missed: int = 3  # ...but only once the current track has missed this many frames
    process_noise: float = 4.0
    measurement_noise: float = 4.0
    trajectory_length: int = 90


def _default_ball_detector() -> DetectorConfig:
    return DetectorConfig(classes=["sports ball"], confidence=0.1, iou=0.3, imgsz=1280)


@dataclass
class BallConfig:
    enabled: bool = True
    # Reuse the player detector's YOLO pass when both use the same weights,
    # imgsz and device and neither uses tiling (roughly halves inference time).
    share_player_inference: bool = True
    detector: DetectorConfig = field(default_factory=_default_ball_detector)
    filter: BallFilterConfig = field(default_factory=BallFilterConfig)
    tracker: BallTrackerConfig = field(default_factory=BallTrackerConfig)


@dataclass
class PitchMaskConfig:
    """HSV range treated as grass. Used only when a filter enables it."""

    hsv_lower: list[int] = field(default_factory=lambda: [30, 40, 40])
    hsv_upper: list[int] = field(default_factory=lambda: [90, 255, 255])
    dilate_px: int = 25


@dataclass
class VisualizationConfig:
    draw_player_boxes: bool = True
    draw_player_ids: bool = True
    draw_player_trajectories: bool = True
    player_trajectory_length: int = 40
    draw_ball: bool = True
    draw_ball_trajectory: bool = True
    show_predicted_ball: bool = True
    draw_hud: bool = True
    line_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class IOConfig:
    output_video: str = "outputs/annotated.mp4"
    output_csv: str = "outputs/tracks.csv"
    output_summary: Optional[str] = "outputs/summary.json"
    fourcc: str = "mp4v"
    fallback_fps: float = 25.0  # used when the container reports no/invalid FPS
    max_frames: Optional[int] = None  # stop early (handy for quick checks)
    log_every: int = 50  # progress log interval in frames
    overwrite: bool = True


@dataclass
class AppConfig:
    io: IOConfig = field(default_factory=IOConfig)
    players: PlayersConfig = field(default_factory=PlayersConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    pitch_mask: PitchMaskConfig = field(default_factory=PitchMaskConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# --------------------------------------------------------------------------- #
# Construction from plain dicts
# --------------------------------------------------------------------------- #
def _unwrap_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _build(cls: type, data: Any, path: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"'{path or 'root'}' must be a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{path or 'root'}': {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(known))}"
        )

    kwargs = {}
    for name, value in data.items():
        tp = _unwrap_optional(hints[name])
        key_path = f"{path}.{name}" if path else name
        if dataclasses.is_dataclass(tp):
            kwargs[name] = _build(tp, value, key_path)
        else:
            kwargs[name] = _coerce(tp, value, key_path, optional=_is_optional(hints[name]))
    return cls(**kwargs)


def _is_optional(tp: Any) -> bool:
    return type(None) in typing.get_args(tp)


def _coerce(tp: Any, value: Any, path: str, optional: bool) -> Any:
    if value is None:
        if optional:
            return None
        raise ConfigError(f"'{path}' may not be null")
    origin = typing.get_origin(tp)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"'{path}' must be a list, got {type(value).__name__}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{path}' must be true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{path}' must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{path}' must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{path}' must be a string, got {value!r}")
        return value
    return value


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _validate_detector(d: DetectorConfig, where: str) -> None:
    _check(bool(d.weights.strip()), f"{where}.weights must not be empty")
    _check(len(d.classes) > 0, f"{where}.classes must list at least one class name")
    _check(all(isinstance(c, str) and c for c in d.classes), f"{where}.classes must be non-empty strings")
    _check(0.0 <= d.confidence <= 1.0, f"{where}.confidence must be in [0, 1]")
    _check(0.0 < d.iou <= 1.0, f"{where}.iou must be in (0, 1]")
    _check(d.imgsz >= 32 and d.imgsz % 32 == 0, f"{where}.imgsz must be a multiple of 32 and >= 32")
    t = d.tiling
    _check(t.tile_size >= 64, f"{where}.tiling.tile_size must be >= 64")
    _check(0.0 <= t.overlap < 0.9, f"{where}.tiling.overlap must be in [0, 0.9)")


def validate(cfg: AppConfig) -> AppConfig:
    io = cfg.io
    _check(len(io.fourcc) == 4, "io.fourcc must be exactly 4 characters (e.g. mp4v, avc1, XVID, MJPG)")
    _check(io.fallback_fps > 0, "io.fallback_fps must be > 0")
    _check(io.max_frames is None or io.max_frames > 0, "io.max_frames must be > 0 or null")
    _check(io.log_every > 0, "io.log_every must be > 0")
    _check(bool(io.output_video) and bool(io.output_csv), "io.output_video and io.output_csv are required")

    _validate_detector(cfg.players.detector, "players.detector")
    pf = cfg.players.filter
    _check(pf.min_height_px >= 0, "players.filter.min_height_px must be >= 0")
    _check(0 < pf.max_height_frac <= 1, "players.filter.max_height_frac must be in (0, 1]")
    _check(0 < pf.min_aspect <= pf.max_aspect, "players.filter: need 0 < min_aspect <= max_aspect")
    pt = cfg.players.tracker
    _check(0 <= pt.track_activation_threshold <= 1, "players.tracker.track_activation_threshold must be in [0, 1]")
    _check(0 <= pt.high_conf_det_threshold <= 1, "players.tracker.high_conf_det_threshold must be in [0, 1]")
    _check(pt.lost_track_buffer >= 0, "players.tracker.lost_track_buffer must be >= 0")
    _check(0 < pt.minimum_iou_threshold <= 1, "players.tracker.minimum_iou_threshold must be in (0, 1]")
    _check(pt.minimum_consecutive_frames >= 1, "players.tracker.minimum_consecutive_frames must be >= 1")

    if cfg.ball.enabled:
        _validate_detector(cfg.ball.detector, "ball.detector")
    bf = cfg.ball.filter
    _check(0 <= bf.min_size_px <= bf.max_size_px, "ball.filter: need 0 <= min_size_px <= max_size_px")
    _check(bf.max_aspect_ratio >= 1, "ball.filter.max_aspect_ratio must be >= 1")
    bt = cfg.ball.tracker
    _check(bt.gate_px > 0, "ball.tracker.gate_px must be > 0")
    _check(bt.gate_growth_px >= 0, "ball.tracker.gate_growth_px must be >= 0")
    _check(bt.max_missed_frames >= 0, "ball.tracker.max_missed_frames must be >= 0")
    _check(bt.min_hits >= 1, "ball.tracker.min_hits must be >= 1")
    _check(0 <= bt.reinit_confidence <= 1, "ball.tracker.reinit_confidence must be in [0, 1]")
    _check(bt.process_noise > 0 and bt.measurement_noise > 0, "ball.tracker noise values must be > 0")
    _check(bt.trajectory_length >= 2, "ball.tracker.trajectory_length must be >= 2")

    pm = cfg.pitch_mask
    _check(len(pm.hsv_lower) == 3 and len(pm.hsv_upper) == 3, "pitch_mask.hsv_lower/hsv_upper need 3 values (H, S, V)")
    _check(pm.dilate_px >= 0, "pitch_mask.dilate_px must be >= 0")

    vz = cfg.visualization
    _check(vz.player_trajectory_length >= 2, "visualization.player_trajectory_length must be >= 2")
    _check(vz.line_thickness >= 1, "visualization.line_thickness must be >= 1")
    _check(vz.font_scale > 0, "visualization.font_scale must be > 0")
    return cfg


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def config_from_dict(data: Optional[dict]) -> AppConfig:
    return validate(_build(AppConfig, data or {}, ""))


def load_config(path: Optional[str | Path]) -> AppConfig:
    """Load and validate a YAML config. ``None`` returns the built-in defaults."""
    if path is None:
        return validate(AppConfig())
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {p}: {exc}") from exc
    return config_from_dict(data)


def config_to_dict(cfg: AppConfig) -> dict:
    return dataclasses.asdict(cfg)
