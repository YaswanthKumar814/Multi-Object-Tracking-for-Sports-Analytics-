"""Exception hierarchy.

Every error the pipeline raises on purpose derives from ``SoccerTrackerError``
so the CLI can print a clean message instead of a traceback, while genuine
bugs still surface as normal exceptions.
"""


class SoccerTrackerError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(SoccerTrackerError):
    """The configuration file is missing, malformed, or has invalid values."""


class VideoError(SoccerTrackerError):
    """Base class for video input/output problems."""


class VideoOpenError(VideoError):
    """The input video is missing, unsupported, or cannot be decoded."""


class VideoWriteError(VideoError):
    """The output video cannot be created or written."""


class ModelLoadError(SoccerTrackerError):
    """Model weights are missing, unavailable, or failed to load."""


class ProcessingError(SoccerTrackerError):
    """A failure while processing frames (wraps the original exception)."""


class DependencyError(SoccerTrackerError):
    """An optional runtime dependency (e.g. ultralytics) is not installed."""
