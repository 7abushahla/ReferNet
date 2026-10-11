"""Design canvas: every document, reference, zone mask and ground-truth mask lives on one grid.

Canvas 512 (H) x 800 (W). Documents are letterboxed with aspect preserved (all our designs are close
to 1.6:1, so padding is a few pixels). SIDTD templates are flat renders of one size per design;
FantasyID images are already document crops with rectangles in shipped coordinates, so rectification
is a similarity transform recorded per image so that rectangles map exactly. A homography path is
kept for sources that ship a document quad.
"""
from __future__ import annotations

import cv2
import numpy as np

CANVAS_H, CANVAS_W = 512, 800
CANVAS_EXT = ".png"  # lossless: nothing is re-encoded after the raw file (spec of 10 Oct 2026)


def letterbox_transform(width: int, height: int) -> tuple[float, int, int]:
    s = min(CANVAS_W / width, CANVAS_H / height)
    nw, nh = round(width * s), round(height * s)
    return s, (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2


def to_canvas(img: np.ndarray, quad: np.ndarray | None = None, mode: str = "letterbox"):
    """Place an image on the canvas. Returns (canvas BGR uint8, valid mask uint8, 3x3 transform)."""
    h, w = img.shape[:2]
    if mode not in ("letterbox", "direct"):
        raise ValueError(f"unknown input mode {mode!r}")
    if mode == "direct":
        if quad is not None:
            raise ValueError("direct resize cannot include a perspective transform")
        resized = cv2.resize(img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
        transform = np.diag([CANVAS_W / w, CANVAS_H / h, 1.])
        return resized, np.full((CANVAS_H, CANVAS_W), 255, np.uint8), transform
    if quad is not None:
        dst = np.float32([[0, 0], [CANVAS_W - 1, 0], [CANVAS_W - 1, CANVAS_H - 1], [0, CANVAS_H - 1]])
        H = cv2.getPerspectiveTransform(np.float32(quad), dst)
        out = cv2.warpPerspective(img, H, (CANVAS_W, CANVAS_H), flags=cv2.INTER_AREA)
        return out, np.full((CANVAS_H, CANVAS_W), 255, np.uint8), H
    s, ox, oy = letterbox_transform(w, h)
    nw, nh = round(w * s), round(h * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    out[oy:oy + nh, ox:ox + nw] = resized
    valid = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    valid[oy:oy + nh, ox:ox + nw] = 255
    T = np.array([[s, 0, ox], [0, s, oy], [0, 0, 1]], dtype=np.float64)
    return out, valid, T


def rect_to_canvas(x: float, y: float, w: float, h: float, T: np.ndarray) -> tuple[int, int, int, int]:
    """Map an axis-aligned rectangle through T to canvas pixel bounds (x0, y0, x1, y1)."""
    pts = np.float64([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]]).T
    q = T @ pts
    q = q[:2] / q[2]
    x0, y0 = np.floor(q.min(axis=1)).astype(int)
    x1, y1 = np.ceil(q.max(axis=1)).astype(int)
    return int(max(0, x0)), int(max(0, y0)), int(min(CANVAS_W, x1)), int(min(CANVAS_H, y1))
