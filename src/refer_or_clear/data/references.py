"""Design-median references and zone masks.

Reference: the pixel-wise median over genuine canvases of a design. Personal content differs across
identities and vanishes in the median; the printed design survives. To keep the query's identity out
of its own reference without a median per query, identities of each design are split into two folds
by a hash, and the reference for a query is the median over the opposite fold. A median over all
identities is also written, for figures only.

Zone mask: the union of all field rectangles over the design's genuine annotations, mapped to the
canvas and dilated a few pixels. It marks where content is allowed to differ from the reference. It is
design-level metadata, identical for genuine and altered queries, never derived from which field was
altered.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .rectify import CANVAS_H, CANVAS_W, rect_to_canvas

ZONE_DILATE_PX = 4


def fold_of(identity_id: str) -> int:
    return int(hashlib.md5(identity_id.encode()).hexdigest(), 16) % 2


def median_canvas(paths: list[Path]) -> np.ndarray:
    stack = np.stack([cv2.imread(str(p)) for p in paths]).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def build_references(records: pd.DataFrame, canvas_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    genuine = records[records.label == 0].copy()
    genuine["fold"] = genuine.identity_id.map(fold_of)
    rows = []
    for design, g in genuine.groupby("design_id"):
        tag = design.replace("/", "_")
        for fold in (0, 1):
            sub = g[g.fold == fold]
            if len(sub):
                med = median_canvas([canvas_dir / f"{i}.jpg" for i in sub.image_id])
                cv2.imwrite(str(out_dir / f"{tag}_fold{fold}.jpg"), med, [cv2.IMWRITE_JPEG_QUALITY, 95])
        med = median_canvas([canvas_dir / f"{i}.jpg" for i in g.image_id])
        cv2.imwrite(str(out_dir / f"{tag}_all.jpg"), med, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(design_id=design, n_genuine=len(g), n_identities=g.identity_id.nunique(),
                         n_fold0=int((g.fold == 0).sum()), n_fold1=int((g.fold == 1).sum())))
    return pd.DataFrame(rows)


def reference_path(ref_dir: Path, design_id: str, query_identity: str) -> Path:
    """The reference for a query: the median over the fold that does not contain its identity."""
    other = 1 - fold_of(query_identity)
    p = ref_dir / f"{design_id.replace('/', '_')}_fold{other}.jpg"
    return p if p.is_file() else ref_dir / f"{design_id.replace('/', '_')}_all.jpg"


_VIA_CACHE: dict[str, dict] = {}


def _sidtd_rects(root: Path, annotation: str, image_name: str):
    key = str(root / annotation)
    if key not in _VIA_CACHE:
        via = json.load(open(key))["_via_img_metadata"]
        _VIA_CACHE[key] = {v["filename"]: v["regions"] for v in via.values()}
    regions = _VIA_CACHE[key].get(image_name, [])
    return [(r["region_attributes"].get("field_name", ""), r["shape_attributes"]["x"], r["shape_attributes"]["y"],
             r["shape_attributes"]["width"], r["shape_attributes"]["height"])
            for r in regions if r["shape_attributes"].get("name") == "rect"]


def _fantasy_rects(root: Path, annotation: str, only_altered: bool = False):
    j = json.load(open(root / annotation))
    out = []
    for r in j.get("regions", []):
        if only_altered and r["region_attributes"].get("region_provenance") != "altered":
            continue
        s = r["shape_attributes"]
        if s.get("name") == "rect":
            out.append((r["region_attributes"].get("field_name", ""), s["x"], s["y"], s["width"], s["height"]))
    return out


def genuine_rects(root: Path, row):
    if row.dataset == "sidtd":
        nn = row.identity_id.rsplit("_", 1)[1]
        return _sidtd_rects(root, row.annotation, f"{nn}.jpg")
    if row.dataset == "fantasyid":
        return _fantasy_rects(root, row.annotation)
    return []


def build_zone_masks(records: pd.DataFrame, root: Path, transforms: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for design, g in records[records.label == 0].groupby("design_id"):
        zone = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
        names = set()
        for r in g.itertuples(index=False):
            if not r.annotation:
                continue
            T = transforms[r.image_id]
            for name, x, y, w, h in genuine_rects(root, r):
                x0, y0, x1, y1 = rect_to_canvas(x, y, w, h, T)
                zone[y0:y1, x0:x1] = 255
                names.add(name)
        if ZONE_DILATE_PX:
            zone = cv2.dilate(zone, np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8))
        cv2.imwrite(str(out_dir / f"{design.replace('/', '_')}.png"), zone)
        rows.append(dict(design_id=design, n_fields=len(names), zone_fraction=round(float(zone.mean() / 255), 3)))
    return pd.DataFrame(rows)
