"""Video reading and writing with explicit validation and resource cleanup.

Both classes are context managers; ``cv2.VideoCapture`` / ``cv2.VideoWriter``
handles are always released, including when an exception escapes the
processing loop.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .errors import VideoOpenError, VideoWriteError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: Optional[int]  # None when the container does not report it reliably
    fps_was_fallback: bool = False

    @property
    def duration_s(self) -> Optional[float]:
        if self.frame_count is None:
            return None
        return self.frame_count / self.fps


class VideoReader:
    """Iterate over ``(frame_index, timestamp_seconds, frame_bgr)``.

    Timestamps are computed as ``frame_index / fps`` rather than read from
    ``CAP_PROP_POS_MSEC``, which is unreliable for many codecs.
    """

    def __init__(self, path: str | Path, fallback_fps: float = 25.0, max_frames: Optional[int] = None):
        self.path = Path(path)
        self.fallback_fps = fallback_fps
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self.info: Optional[VideoInfo] = None
        self._first_frame: Optional[np.ndarray] = None

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> "VideoReader":
        validate_input_path(self.path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(
                f"OpenCV could not open '{self.path}'. The file may be corrupt or use a codec "
                "your OpenCV build cannot decode. Try re-encoding it, e.g.: "
                "ffmpeg -i input.ext -c:v libx264 -pix_fmt yuv420p output.mp4"
            )
        self._cap = cap

        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            self.release()
            raise VideoOpenError(
                f"'{self.path}' opened but no frame could be decoded. The file is empty, "
                "truncated, or uses an unsupported codec."
            )
        self._first_frame = frame
        height, width = frame.shape[:2]

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_fallback = False
        if not fps or math.isnan(fps) or fps <= 0 or fps > 1000:
            log.warning("Video reports invalid FPS (%r); using fallback %.2f", fps, self.fallback_fps)
            fps = self.fallback_fps
            fps_fallback = True

        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = count if count > 0 else None

        self.info = VideoInfo(self.path, width, height, float(fps), frame_count, fps_fallback)
        log.info(
            "Opened %s: %dx%d @ %.2f FPS, %s frames",
            self.path.name, width, height, fps, frame_count if frame_count is not None else "unknown",
        )
        return self

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

    # -- iteration --------------------------------------------------------- #
    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if self._cap is None or self.info is None:
            raise VideoOpenError("VideoReader.frames() called before open()")
        idx = 0
        frame = self._first_frame
        self._first_frame = None
        while frame is not None:
            if self.max_frames is not None and idx >= self.max_frames:
                return
            yield idx, idx / self.info.fps, frame
            idx += 1
            ok, frame = self._cap.read()
            if not ok:
                frame = None
        if self.info.frame_count is not None and idx < self.info.frame_count - 1 and (
            self.max_frames is None or idx < self.max_frames
        ):
            # Containers often over-report by one frame; only warn on a real shortfall.
            log.warning(
                "Decoding stopped at frame %d of a reported %d. The file may be truncated.",
                idx, self.info.frame_count,
            )


class VideoWriter:
    """Write BGR frames of a fixed size to a video file."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], fourcc: str = "mp4v", overwrite: bool = True):
        self.path = Path(path)
        self.fps = fps
        self.size = size  # (width, height)
        self.fourcc = fourcc
        self.overwrite = overwrite
        self._writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0

    def open(self) -> "VideoWriter":
        if self.path.exists() and not self.overwrite:
            raise VideoWriteError(f"Output '{self.path}' already exists and overwrite is disabled.")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoWriteError(f"Cannot create output directory '{self.path.parent}': {exc}") from exc
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self.size)
        if not writer.isOpened():
            writer.release()
            raise VideoWriteError(
                f"Could not open a video writer for '{self.path}' with codec '{self.fourcc}'. "
                "Check that the codec matches the extension (mp4v/avc1 -> .mp4, XVID/MJPG -> .avi) "
                "and that your OpenCV build supports it."
            )
        self._writer = writer
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise VideoWriteError("VideoWriter.write() called before open()")
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise VideoWriteError(f"Frame size {w}x{h} does not match writer size {self.size[0]}x{self.size[1]}")
        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def validate_input_path(path: Path) -> None:
    if not path.exists():
        raise VideoOpenError(f"Input video not found: '{path}'")
    if not path.is_file():
        raise VideoOpenError(f"Input path is not a file: '{path}'")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoOpenError(
            f"Unsupported video extension '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if path.stat().st_size == 0:
        raise VideoOpenError(f"Input video is empty (0 bytes): '{path}'")
