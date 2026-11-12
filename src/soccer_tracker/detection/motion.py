"""Motion-based ball candidates for when the appearance detector goes blind.

A ball struck hard is motion-blurred into a pale streak, and COCO "sports
ball" detectors rarely fire on it, which is exactly when tracking matters
most (shots, long passes). The ball in flight is, however, the most obvious
small, bright, *moving* thing in the frame. This module finds such blobs:

1. Estimate the camera motion between frames (ORB features + RANSAC
   similarity transform) and warp the two previous frames onto the current
   one, so a panning or zooming camera does not make the whole frame "move".
2. Three-frame differencing using only past frames: a pixel belongs to a
   moving object *now* if it differs from both t-1 and t-2. The ghost left at
   the object's old position differs only from t-1, so it is suppressed.
3. Keep bright, low-saturation pixels (a white/pale ball) and return small,
   roughly compact connected components as candidates.

If too much of the frame is "moving" after compensation (``max_moving_fraction``),
the camera model has failed (cut, dissolve, fast zoom) and no candidates are
returned for that frame.

The result is deliberately recall-oriented and noisy (moving boots, shirt
numbers, crowd), so single candidates are never used as the ball. They are
linked over time into flights (tracking/flight.py), and only a fast, smooth
flight that starts where the ball was is adopted by the ball tracker.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np

from ..config import MotionBallConfig
from .base import Detections



class MotionBallDetector:
    """Stateful: call ``detect`` once per frame, in order."""

    def __init__(self, cfg: MotionBallConfig):
        self.cfg = cfg
        self._orb = cv2.ORB_create(cfg.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Each entry: (gray frame, keypoints, descriptors, affine mapping this frame -> the next one)
        self._history: deque[list] = deque(maxlen=2)
        self._kernel = np.ones((3, 3), np.uint8)

    def reset(self) -> None:
        self._history.clear()

    def _features(self, gray: np.ndarray):
        return self._orb.detectAndCompute(gray, None)

    def _affine(self, kp_a, des_a, kp_b, des_b) -> Optional[np.ndarray]:
        """2x3 similarity transform mapping frame a onto frame b, or None."""
        if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return None
        matches = self._matcher.match(des_a, des_b)
        if len(matches) < 10:
            return None
        pa = np.float32([kp_a[m.queryIdx].pt for m in matches])
        pb = np.float32([kp_b[m.trainIdx].pt for m in matches])
        A, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if A is None or inliers is None or int(inliers.sum()) < 10:
            return None
        return A

    @staticmethod
    def _compose(second: np.ndarray, first: np.ndarray) -> np.ndarray:
        """Affine that applies ``first`` then ``second``."""
        m1 = np.vstack([first, [0, 0, 1]])
        m2 = np.vstack([second, [0, 0, 1]])
        return (m2 @ m1)[:2]

    def detect(self, frame: np.ndarray) -> Detections:
        cfg = self.cfg
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._features(gray)

        if len(self._history) == 2:
            prev2, prev1 = self._history[0], self._history[1]
            a1 = self._affine(prev1[1], prev1[2], kp, des)  # t-1 -> t
            prev1[3] = a1
            a2 = self._compose(a1, prev2[3]) if (a1 is not None and prev2[3] is not None) else None  # t-2 -> t
            dets = self._candidates(frame, gray, prev1[0], prev2[0], a1, a2)
        else:
            if len(self._history) == 1:
                prev1 = self._history[0]
                prev1[3] = self._affine(prev1[1], prev1[2], kp, des)
            dets = Detections.empty()

        self._history.append([gray, kp, des, None])
        return dets

    def _candidates(self, frame, gray, g1, g2, a1, a2) -> Detections:
        cfg = self.cfg
        if a1 is None or a2 is None:
            return Detections.empty()  # camera motion unknown (cut, blur): differencing would be noise
        h, w = gray.shape
        flags = cv2.INTER_LINEAR
        w1 = cv2.warpAffine(g1, a1, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        w2 = cv2.warpAffine(g2, a2, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        cur = gray.astype(np.int16)
        moving = (np.abs(cur - w1) > cfg.diff_threshold) & (np.abs(cur - w2) > cfg.diff_threshold)
        # After a good camera-motion compensation only the players and the ball
        # move: about 1% of the frame. Much more means the similarity model does
        # not hold (fast zoom with parallax, a cut, a replay dissolve), and any
        # "motion" is alignment error: report nothing rather than noise.
        if float(moving.mean()) > cfg.max_moving_fraction:
            return Detections.empty()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright = (hsv[..., 2] >= cfg.min_value) & (hsv[..., 1] <= cfg.max_saturation)
        mask = (moving & bright).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            side = max(bw, bh)
            if not (cfg.min_area_px <= area <= cfg.max_area_px and side <= cfg.max_side_px):
                continue
            if side / max(1, min(bw, bh)) > cfg.max_aspect_ratio:
                continue
            boxes.append([x, y, x + bw, y + bh])
        if not boxes:
            return Detections.empty()
        k = len(boxes)
        return Detections(np.array(boxes, float), np.full(k, cfg.confidence), np.array(["ball (motion)"] * k, dtype=object))

