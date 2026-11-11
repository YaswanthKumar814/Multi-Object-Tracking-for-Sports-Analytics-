"""Build a multi-object tracker by name.

All three algorithms come from Roboflow's ``trackers`` package (Apache-2.0)
and share one ``update(sv.Detections) -> sv.Detections`` interface, so they are
interchangeable behind :class:`~soccer_tracker.tracking.PlayerTracker`.

==========  ==================================================================
``sort``      SORT (Bewley et al., 2016): Kalman prediction + one IoU matching
              stage. Every detection may match an existing track; unmatched
              detections above ``track_activation_threshold`` start new tracks.
``ocsort``    OC-SORT (Cao et al., 2023): SORT with observation-centric
              re-update, momentum and recovery. Only detections above
              ``high_conf_det_threshold`` are used.
``bytetrack`` ByteTrack (Zhang et al., 2022): matches high-confidence
              detections first, then uses the low-confidence ones to keep
              existing tracks alive. This is the proposed player tracker.
==========  ==================================================================

The five parameters below are shared so the comparison differs only in the
association algorithm; a tracker simply ignores a parameter it does not have.
"""

from __future__ import annotations

from typing import Any

from ..errors import ConfigError, DependencyError

ALGORITHMS = ("sort", "ocsort", "bytetrack")

# Names used consistently in tables, plots, the report and the slides.
TRACKER_LABELS = {
    "sort": "YOLO11n + SORT",
    "ocsort": "YOLO11n + OC-SORT",
    "bytetrack": "YOLO11n + ByteTrack (proposed)",
}
SHORT_LABELS = {"sort": "SORT", "ocsort": "OC-SORT", "bytetrack": "ByteTrack (proposed)"}

SHARED_PARAMS = (
    "lost_track_buffer",
    "minimum_consecutive_frames",
    "minimum_iou_threshold",
    "track_activation_threshold",
    "high_conf_det_threshold",
)
_ACCEPTED = {
    "sort": ("lost_track_buffer", "minimum_consecutive_frames", "minimum_iou_threshold",
             "track_activation_threshold"),
    "ocsort": ("lost_track_buffer", "minimum_consecutive_frames", "minimum_iou_threshold",
               "high_conf_det_threshold"),
    "bytetrack": SHARED_PARAMS,
}


def accepted_params(algorithm: str) -> tuple[str, ...]:
    return _ACCEPTED[algorithm]


def build_mot_tracker(algorithm: str, frame_rate: float, **params: Any):
    """Create a tracker. ``params`` may contain any of :data:`SHARED_PARAMS`."""
    if algorithm not in ALGORITHMS:
        raise ConfigError(f"Unknown tracker '{algorithm}'. Choose one of: {', '.join(ALGORITHMS)}")
    unknown = sorted(set(params) - set(SHARED_PARAMS))
    if unknown:
        raise ConfigError(f"Unknown tracker parameter(s): {', '.join(unknown)}")
    try:
        from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker
    except ImportError as exc:
        raise DependencyError(
            "Player tracking needs the 'trackers' and 'supervision' packages. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    cls = {"sort": SORTTracker, "ocsort": OCSORTTracker, "bytetrack": ByteTrackTracker}[algorithm]
    kwargs = {k: v for k, v in params.items() if k in _ACCEPTED[algorithm]}
    return cls(frame_rate=float(frame_rate), **kwargs)
