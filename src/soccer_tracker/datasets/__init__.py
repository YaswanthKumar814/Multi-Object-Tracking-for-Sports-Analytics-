"""Dataset adapters: one module per dataset, one common in-memory representation."""

from .base import DatasetError, Sequence, SequenceInfo
from .infactory import InFactoryClip, list_clips
from .sportsmot import SportsMOTSequence
from .uvy_track import UVYTrackSequence

__all__ = [
    "DatasetError",
    "InFactoryClip",
    "Sequence",
    "SequenceInfo",
    "SportsMOTSequence",
    "UVYTrackSequence",
    "list_clips",
]
