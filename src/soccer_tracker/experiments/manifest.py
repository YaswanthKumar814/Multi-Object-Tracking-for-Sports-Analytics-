"""Load ``configs/experiments.yaml`` and build the sequences it names."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from ..config import AppConfig
from ..datasets import InFactoryClip, Sequence, SportsMOTSequence, UVYTrackSequence, list_clips
from ..datasets.base import DatasetError


@dataclass
class Manifest:
    raw: dict[str, Any]
    results_dir: Path
    skipped: list[str] = field(default_factory=list)  # datasets/sequences that could not be loaded

    # -- detector --------------------------------------------------------- #
    def app_config(self, weights: Optional[str] = None) -> AppConfig:
        """Project defaults with the manifest's detector settings applied."""
        det = self.raw.get("detector", {})
        cfg = AppConfig()
        for d in (cfg.players.detector, cfg.ball.detector):
            d.weights = weights or det.get("weights", d.weights)
            d.imgsz = int(det.get("imgsz", d.imgsz))
            d.device = det.get("device", d.device)
        return cfg

    def detector_tag(self, weights: Optional[str] = None) -> str:
        cfg = self.app_config(weights)
        return f"{Path(cfg.players.detector.weights).stem}_{cfg.players.detector.imgsz}"

    # -- sequences -------------------------------------------------------- #
    def _try(self, label: str, build) -> Optional[Sequence]:
        try:
            return build()
        except DatasetError as exc:
            self.skipped.append(f"{label}: {exc}")
            return None

    def sportsmot(self, role: str) -> list[Sequence]:
        sec = self.raw.get("sportsmot") or {}
        out = []
        for name in sec.get(role, []) or []:
            seq = self._try(f"SportsMOT/{name}", lambda n=name: SportsMOTSequence(
                sec["root"], n, split=sec.get("split", "val"), role=role))
            if seq:
                out.append(seq)
        return out

    def uvy_track(self) -> list[Sequence]:
        sec = self.raw.get("uvy_track") or {}
        out = []
        for item in sec.get("test", []) or []:
            seq = self._try(f"UVY-Track/{item['name']}", lambda it=item: UVYTrackSequence(
                sec["root"], it["name"], it.get("first_frame"), it.get("last_frame"), role="test"))
            if seq:
                out.append(seq)
        return out

    def infactory(self, only: Optional[list[str]] = None) -> list[Sequence]:
        sec = self.raw.get("infactory") or {}
        if not sec:
            return []
        try:
            clips = list_clips(sec["root"]) if sec.get("clips", "all") == "all" else list(sec["clips"])
        except DatasetError as exc:
            self.skipped.append(f"InFactory: {exc}")
            return []
        if only is not None:
            clips = [c for c in clips if c in only]
        out = []
        for clip in clips:
            seq = self._try(f"InFactory/{clip}", lambda c=clip: InFactoryClip(sec["root"], c))
            if seq:
                out.append(seq)
        return out

    def player_tracking_sequences(self) -> list[Sequence]:
        return self.sportsmot("dev") + self.sportsmot("test") + self.uvy_track()

    def ball_sequences(self) -> list[Sequence]:
        return self.uvy_track() + self.infactory()

    def all_sequences(self) -> list[Sequence]:
        return self.sportsmot("dev") + self.sportsmot("test") + self.uvy_track() + self.infactory()


def load_manifest(path: str | Path = "configs/experiments.yaml") -> Manifest:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Manifest(raw=raw, results_dir=Path(raw.get("results_dir", "results")))
