"""Register every image of a design to the design itself, so that the median reference is sharp and
the query and reference line up pixel for pixel.

Why: SIDTD templates are flat renders with one framing per design, but FantasyID images are phone
photos of printed cards cropped by the dataset authors; the crop leaves a residual rotation of about
one degree and a shift of 10 to 20 px on our canvas, enough to blur every printed line in a median.
The golden-sample inspection literature registers the query to the template before differencing;
we do the same, with no annotations, so the step is deployable as is.

How: SIFT keypoints, Lowe ratio test, RANSAC similarity (rotation, scale, translation). Two passes
per design: pass 1 registers the genuine images to one anchor and takes a median; pass 2 registers
every image of the design (genuine and altered) to that median. An image whose inlier count is below
MIN_INLIERS keeps the identity transform and is flagged. The returned 3x3 matrix maps letterbox
canvas coordinates to registered canvas coordinates, so it composes with the letterbox transform and
every rectangle moves with its image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .rectify import CANVAS_H, CANVAS_W

MIN_INLIERS = 40
RATIO = 0.75
RANSAC_PX = 3.0
N_FEATURES = 3000


def _features(img: np.ndarray):
    sift = cv2.SIFT_create(N_FEATURES)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def estimate_similarity(src_feats, dst_feats) -> tuple[np.ndarray | None, int]:
    """3x3 similarity mapping src pixels to dst pixels, and the RANSAC inlier count."""
    k1, d1 = src_feats
    k2, d2 = dst_feats
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < RATIO * b.distance]
    if len(good) < 12:
        return None, len(good)
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if A is None or inl is None:
        return None, 0
    return np.vstack([A, [0.0, 0.0, 1.0]]), int(inl.sum())


def _params(R: np.ndarray) -> dict:
    return dict(rot_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), scale=float(np.hypot(R[0, 0], R[1, 0])),
                shift_px=float(np.hypot(R[0, 2], R[1, 2])))


def warp(img: np.ndarray, R: np.ndarray, border: int = 0) -> np.ndarray:
    return cv2.warpAffine(img, R[:2], (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def register_design(design_id: str, image_ids: list[str], genuine_ids: list[str], canvas_dir: Path) -> pd.DataFrame:
    """Estimate one similarity per image of a design from its letterbox canvases.

    Returns rows (image_id, applied, inliers, rot_deg, scale, shift_px, R) with R as a 3x3 list.
    """
    imgs = {i: cv2.imread(str(canvas_dir / f"{i}.jpg")) for i in image_ids}
    feats = {i: _features(im) for i, im in imgs.items()}
    anchor = sorted(genuine_ids)[0]
    # pass 1: genuine images to the anchor, then a median
    stack = []
    for i in genuine_ids:
        R, n = estimate_similarity(feats[i], feats[anchor])
        stack.append(warp(imgs[i], R) if R is not None and n >= MIN_INLIERS else imgs[i])
    target = np.median(np.stack(stack), axis=0).astype(np.uint8)
    target_feats = _features(target)
    # pass 2: every image to the median
    rows = []
    for i in image_ids:
        R, n = estimate_similarity(feats[i], target_feats)
        applied = R is not None and n >= MIN_INLIERS
        if not applied:
            R = np.eye(3)
        rows.append(dict(design_id=design_id, image_id=i, applied=bool(applied), inliers=int(n), **_params(R), R=R.tolist()))
    return pd.DataFrame(rows)
