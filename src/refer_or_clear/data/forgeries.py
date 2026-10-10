"""Ground-truth masks for altered images, as filled rectangles on the canvas.

SIDTD: the altered field is named in the forgery's JSON and the forgery keeps the geometry of its
source template, so the mask is that field's rectangle from the source's VIA annotation, dilated 8 px
for the unrecorded shift of crop-and-replace. FantasyID: the rectangles marked 'altered' in the
image's own JSON (face swaps mark the portrait and the ghost image). A box mask, not an exact pixel
mask; the paper says so. Masks are written only for altered images.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .rectify import CANVAS_H, CANVAS_W, rect_to_canvas
from .references import _fantasy_rects, _sidtd_rects

SIDTD_DILATE_PX = 8


def altered_rects(root: Path, row):
    if row.dataset == "sidtd":
        src = row.source_id.split("/", 1)[1]
        design, nn = src.rsplit("_", 1)
        rects = _sidtd_rects(root, f"sidtd/templates/Annotations/reals/{design}.json", f"{nn}.jpg")
        wanted = set(row.altered_fields.split(",")) if row.altered_fields else set()
        return [r for r in rects if r[0] in wanted]
    if row.dataset == "fantasyid" and row.annotation:
        return _fantasy_rects(root, row.annotation, only_altered=True)
    return []


def build_masks(records: pd.DataFrame, root: Path, transforms: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records[records.label == 1].itertuples(index=False):
        mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
        T = transforms[r.image_id]
        rects = altered_rects(root, r)
        for _, x, y, w, h in rects:
            x0, y0, x1, y1 = rect_to_canvas(x, y, w, h, T)
            mask[y0:y1, x0:x1] = 255
        if r.dataset == "sidtd" and SIDTD_DILATE_PX:
            mask = cv2.dilate(mask, np.ones((2 * SIDTD_DILATE_PX + 1,) * 2, np.uint8))
        cv2.imwrite(str(out_dir / f"{r.image_id}.png"), mask)
        rows.append(dict(image_id=r.image_id, dataset=r.dataset, n_altered_rects=len(rects),
                         mask_fraction=float(mask.mean() / 255)))
    return pd.DataFrame(rows)
