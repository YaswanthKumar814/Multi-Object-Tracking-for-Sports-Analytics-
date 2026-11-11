"""SportsMOT adapter (Cui et al., ICCV 2023; https://github.com/MCG-NJU/SportsMOT).

Layout on disk (MOT Challenge style, verified on the downloaded files)::

    <root>/<split>/<sequence>/seqinfo.ini      name, frameRate, seqLength, imWidth, imHeight
    <root>/<split>/<sequence>/img1/000001.jpg  1-based, zero-padded frame images
    <root>/<split>/<sequence>/gt/gt.txt        frame, id, x, y, w, h, 1, 1, 1

Only players on the pitch are annotated: referees, coaches and spectators have
no boxes, and there are **no ball annotations**. The dataset is therefore used
for player tracking only.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .base import DatasetError, Sequence, SequenceInfo


class SportsMOTSequence(Sequence):
    has_player_ids = True

    def __init__(self, root: str | Path, name: str, split: str = "val", role: str = "test",
                 max_frames: Optional[int] = None):
        self.dir = Path(root) / split / name
        ini = self.dir / "seqinfo.ini"
        if not ini.is_file():
            raise DatasetError(f"SportsMOT sequence not found: {self.dir} (missing seqinfo.ini)")
        cp = configparser.ConfigParser()
        cp.read(ini, encoding="utf-8")
        s = cp["Sequence"]
        self.img_dir = self.dir / s.get("imDir", "img1")
        self.ext = s.get("imExt", ".jpg")
        length = int(s["seqLength"])
        if max_frames:
            length = min(length, max_frames)
        self.info = SequenceInfo(
            dataset="SportsMOT", name=name, split=split, role=role, fps=float(s["frameRate"]),
            width=int(s["imWidth"]), height=int(s["imHeight"]), num_frames=length,
            source_first_frame=1, source=str(self.img_dir),
        )

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for f in range(1, self.info.num_frames + 1):
            path = self.img_dir / f"{f:06d}{self.ext}"
            img = cv2.imread(str(path))
            if img is None:
                raise DatasetError(f"Could not read frame {path}")
            yield f, img

    def player_gt(self) -> np.ndarray:
        path = self.dir / "gt" / "gt.txt"
        if not path.is_file():
            raise DatasetError(f"SportsMOT ground truth not found: {path}")
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
        rows = rows[(rows[:, 6] != 0) & (rows[:, 0] <= self.info.num_frames)]
        return rows[:, :6].astype(np.float64)
