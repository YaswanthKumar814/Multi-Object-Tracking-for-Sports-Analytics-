"""Ultralytics YOLO detector.

PyTorch and Ultralytics are imported lazily, so the rest of the package
(and the unit tests) work without them.

Licence note: Ultralytics and its official YOLO weights are AGPL-3.0.
See README "Model weights and licensing".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import DetectorConfig
from ..errors import DependencyError, ModelLoadError
from .base import Detections, compute_tiles, nms

log = logging.getLogger(__name__)


class ModelCache:
    """Load each weights file once, even if several detectors use it."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def get(self, weights: str, allow_download: bool) -> Any:
        key = str(Path(weights).expanduser())
        if key not in self._models:
            self._models[key] = load_yolo(key, allow_download)
        return self._models[key]


def _official_asset_names() -> set[str]:
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        return set(GITHUB_ASSETS_NAMES)
    except Exception:  # pragma: no cover - depends on ultralytics internals
        return set()


def load_yolo(weights: str, allow_download: bool = True) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DependencyError(
            "The 'ultralytics' package is required for YOLO detection. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    path = Path(weights).expanduser()
    if not path.is_file():
        official = path.name in _official_asset_names()
        if not official:
            raise ModelLoadError(
                f"Model weights not found: '{path}'. This is not an official Ultralytics asset, "
                "so it cannot be downloaded automatically. Place the .pt file at that path or "
                "update 'weights' in your config. See README section 'Model weights'."
            )
        if not allow_download:
            raise ModelLoadError(
                f"Model weights not found: '{path}' and allow_download is false. Either set "
                f"allow_download: true or download '{path.name}' manually from "
                "https://github.com/ultralytics/assets/releases and place it at that path."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Weights '%s' not found locally; Ultralytics will download them to '%s'", path.name, path)

    try:
        model = YOLO(str(path))
    except Exception as exc:  # Ultralytics raises a variety of exception types
        raise ModelLoadError(
            f"Failed to load model weights '{path}': {exc}. If this was a download, check your "
            "network connection or download the file manually."
        ) from exc
    if getattr(model, "task", "detect") != "detect":
        raise ModelLoadError(f"'{path}' is a '{model.task}' model; an object detection ('detect') model is required.")
    return model


class YoloDetector:
    """Runs a YOLO model and keeps only the configured class names."""

    def __init__(self, cfg: DetectorConfig, cache: Optional[ModelCache] = None):
        self.cfg = cfg
        cache = cache or ModelCache()
        self.model = cache.get(cfg.weights, cfg.allow_download)
        names: dict[int, str] = dict(self.model.names)
        lookup = {v.lower(): k for k, v in names.items()}
        missing = [c for c in cfg.classes if c.lower() not in lookup]
        if missing:
            preview = ", ".join(sorted(names.values())[:30])
            raise ModelLoadError(
                f"Class name(s) {missing} not found in model '{cfg.weights}'. "
                f"Available classes include: {preview}"
            )
        self.class_ids = [lookup[c.lower()] for c in cfg.classes]
        self.id_to_name = names
        self._half = bool(cfg.half)

    def _predict(self, images: list[np.ndarray], imgsz: int) -> list[Detections]:
        kwargs = dict(
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=imgsz,
            classes=self.class_ids,
            device=self.cfg.device,
            verbose=False,
        )
        if self._half:  # only pass when requested; newer Ultralytics warns that 'half' is deprecated
            kwargs["half"] = True
        results = self.model.predict(images, **kwargs)
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(Detections.empty())
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            out.append(Detections(xyxy, conf, np.array([self.id_to_name[int(c)] for c in cls], dtype=object)))
        return out

    def detect(self, frame: np.ndarray) -> Detections:
        if not self.cfg.tiling.enabled:
            return self._predict([frame], self.cfg.imgsz)[0]

        t = self.cfg.tiling
        h, w = frame.shape[:2]
        tiles = compute_tiles(w, h, t.tile_size, t.overlap)
        crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]
        parts = []
        for (x1, y1, _, _), dets in zip(tiles, self._predict(crops, t.tile_size)):
            if len(dets):
                dets.xyxy += np.array([x1, y1, x1, y1], dtype=np.float32)
                parts.append(dets)
        if t.include_full_frame:
            parts.append(self._predict([frame], self.cfg.imgsz)[0])
        return nms(Detections.concat(parts), self.cfg.iou)


# --------------------------------------------------------------------------- #
# One inference pass shared by the player and ball detectors
# --------------------------------------------------------------------------- #
def can_share_inference(a: DetectorConfig, b: DetectorConfig) -> bool:
    """True when two detector configs would run the same network on the same input."""
    return (
        Path(a.weights).expanduser() == Path(b.weights).expanduser()
        and a.imgsz == b.imgsz
        and a.device == b.device
        and a.half == b.half
        and not a.tiling.enabled
        and not b.tiling.enabled
    )


class _SharedPass:
    def __init__(self, detector: YoloDetector):
        self.detector = detector
        self._frame: Optional[np.ndarray] = None
        self._dets: Optional[Detections] = None

    def run(self, frame: np.ndarray) -> Detections:
        # Holding a reference to the last frame guarantees the identity check
        # cannot be fooled by Python re-using an object id.
        if frame is not self._frame:
            self._dets = self.detector.detect(frame)
            self._frame = frame
        assert self._dets is not None
        return self._dets


class ClassView:
    """Detector that returns its classes (at its own threshold) from a shared pass."""

    def __init__(self, shared: _SharedPass, classes: list[str], confidence: float):
        self._shared = shared
        self._classes = {c.lower() for c in classes}
        self._conf = confidence

    def detect(self, frame: np.ndarray) -> Detections:
        d = self._shared.run(frame)
        if len(d) == 0:
            return d
        keep = np.array([str(c).lower() in self._classes for c in d.class_name]) & (d.confidence >= self._conf)
        return d.subset(keep)


def build_shared_detectors(player_cfg: DetectorConfig, ball_cfg: DetectorConfig, cache: Optional[ModelCache] = None):
    """One YOLO pass per frame serving both players and ball (about 2x faster).

    The combined pass uses the lower of the two confidence thresholds and the
    player IoU; each view then applies its own confidence threshold.
    """
    import copy

    merged = copy.deepcopy(player_cfg)
    merged.classes = list(dict.fromkeys(player_cfg.classes + ball_cfg.classes))
    merged.confidence = min(player_cfg.confidence, ball_cfg.confidence)
    shared = _SharedPass(YoloDetector(merged, cache))
    return (
        ClassView(shared, player_cfg.classes, player_cfg.confidence),
        ClassView(shared, ball_cfg.classes, ball_cfg.confidence),
    )
