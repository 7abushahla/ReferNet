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

from .rectify import CANVAS_EXT, CANVAS_H, CANVAS_W, rect_to_canvas

ZONE_DILATE_PX = 4


def fold_of(key: str) -> int:
    """Fold of a provenance component (or any key string): md5 parity. Splits and references use the
    component id, so a crop-and-replace donor never ends up in the reference of its own forgery."""
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 2


def median_canvas(paths: list[Path], valid_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Validity-aware per-pixel median. Padded or warped-out pixels (valid/<id>.png == 0) are excluded;
    returns (median uint8, coverage uint8 = number of valid observations per pixel, capped at 255)."""
    stack = np.stack([cv2.imread(str(p)) for p in paths]).astype(np.float32)
    if valid_dir is not None:
        for k, p in enumerate(paths):
            v = valid_dir / f"{p.stem}.png"
            if v.is_file():
                stack[k][cv2.imread(str(v), cv2.IMREAD_GRAYSCALE) < 128] = np.nan
    cover = np.minimum(np.sum(~np.isnan(stack[..., 0]), axis=0), 255).astype(np.uint8)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    med = np.where(np.isnan(med), 0, med).astype(np.uint8)
    return med, cover


def build_references(records: pd.DataFrame, canvas_dir: Path, out_dir: Path, group: str = "",
                     write_all: bool = True) -> pd.DataFrame:
    """Fold medians per design from the genuine rows of `records`. With `group`, files are named
    <design>_<group>_fold{0,1}.png and both halves must be non-empty (no fallback to the global median)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    genuine = records[records.label == 0].copy()
    genuine["fold"] = genuine.component_id.map(fold_of)
    valid_dir = canvas_dir.parent / "valid"
    rows = []
    for design, g in genuine.groupby("design_id"):
        tag = design.replace("/", "_") + (f"_{group}" if group else "")
        for fold in (0, 1):
            sub = g[g.fold == fold]
            if not len(sub):
                if group:
                    raise ValueError(f"design {design}, group {group}: no genuine image in half {fold}")
                continue
            med, cover = median_canvas([canvas_dir / f"{i}{CANVAS_EXT}" for i in sub.image_id], valid_dir)
            cv2.imwrite(str(out_dir / f"{tag}_fold{fold}.png"), med)
            cv2.imwrite(str(out_dir / f"{tag}_fold{fold}_coverage.png"), cover)
        if write_all:
            med, cover = median_canvas([canvas_dir / f"{i}{CANVAS_EXT}" for i in g.image_id], valid_dir)
            cv2.imwrite(str(out_dir / f"{tag}_all.png"), med)
        rows.append(dict(design_id=design, group=group, n_genuine=len(g), n_identities=g.identity_id.nunique(),
                         n_components=g.component_id.nunique(), n_fold0=int((g.fold == 0).sum()),
                         n_fold1=int((g.fold == 1).sum()), min_coverage=int(cover.min())))
    return pd.DataFrame(rows)


def reference_path(ref_dir: Path, design_id: str, component_id: str, kind: str = "median") -> Path:
    """The reference for a query: the fold median built without its provenance component, or the blank
    specimen registered to that fold. Raises when the file is missing; there is no silent fallback."""
    other = 1 - fold_of(component_id)
    tag = design_id.replace("/", "_")
    p = ref_dir / (f"{tag}_fold{other}.png" if kind == "median" else f"{tag}_specimen_fold{other}.png")
    if not p.is_file():
        raise FileNotFoundError(f"reference {p} missing: design {design_id}, component {component_id}, kind {kind}")
    return p


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


# --- blank specimens -------------------------------------------------------------------------
# Twenty blank design templates reproduced from the IDNet paper figure (data/raw/idnet_templates).
# Ten of them are the MIDV-2020 / SIDTD designs; two (AZ, WV) are the Protocol C designs. A specimen
# is registered to the design median (SIFT + RANSAC similarity) and written as <design>_specimen.png,
# so the reference ablation can swap the median for a true blank without touching anything else.

SPECIMEN_DESIGNS = {  # design_id -> template file stem
    **{f"sidtd/{d}": d for d in ("alb_id", "aze_passport", "esp_id", "est_id", "fin_id", "grc_passport",
                                 "lva_passport", "rus_internalpassport", "srb_passport", "svk_id")},
    "idnet/AZ": "us_az_dl", "idnet/WV": "us_wv_dl",
}


def build_specimens(records: pd.DataFrame, templates_dir: Path, ref_dir: Path) -> pd.DataFrame:
    from .rectify import to_canvas
    from .register import MIN_INLIERS, _features, estimate_similarity, warp
    rows = []
    for design in sorted(records.design_id.unique()):
        stem = SPECIMEN_DESIGNS.get(design)
        tag = design.replace("/", "_")
        median_path = ref_dir / f"{tag}_all.png"
        if stem is None or not (templates_dir / f"{stem}.jpg").is_file() or not median_path.is_file():
            continue
        tpl, _, _ = to_canvas(cv2.imread(str(templates_dir / f"{stem}.jpg")))
        tf = _features(tpl)
        for fold in (0, 1, "all"):
            mp = ref_dir / (f"{tag}_fold{fold}.png" if fold != "all" else f"{tag}_all.png")
            if not mp.is_file():
                continue
            med = cv2.imread(str(mp))
            R, n = estimate_similarity(tf, _features(med))
            ok = R is not None and n >= MIN_INLIERS
            if not ok:
                raise RuntimeError(f"specimen {stem} failed to register to {mp.name} ({n} inliers)")
            spec = warp(tpl, R)
            out = ref_dir / (f"{tag}_specimen_fold{fold}.png" if fold != "all" else f"{tag}_specimen.png")
            cv2.imwrite(str(out), spec)
            rows.append(dict(design_id=design, template=stem, fold=str(fold), inliers=int(n),
                             rot_deg=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 2),
                             scale=round(float(np.hypot(R[0, 0], R[1, 0])), 3),
                             mean_absdiff_to_median=round(float(cv2.absdiff(spec, med).mean()), 1)))
    return pd.DataFrame(rows)


def specimen_path(ref_dir: Path, design_id: str, component_id: str) -> Path:
    return reference_path(ref_dir, design_id, component_id, kind="specimen")
