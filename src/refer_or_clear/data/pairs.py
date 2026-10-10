"""Training and test samples as (query canvas, reference, zone mask, label, ground-truth mask).

One dict per sample with the file paths the dataset needs. The reference for a query never contains
the query's own provenance component: `median` uses the design median of the opposite fold, `single`
draws one genuine canvas of the same design from another component, `specimen` uses the registered
blank template. Wrong-design pairs (train only) are added as hard negatives labelled altered, so the
network fails safe when the reference does not match the document.

Dict keys
    image_id, design_id, path, valid_path (or None), mask_path (or None), zone_path, ref_path,
    label (0 genuine, 1 altered), subset, attack_family, component_id, wrong_design (bool)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rectify import CANVAS_EXT
from .references import fold_of

REFERENCES = ("median", "specimen", "single")


def _tag(design_id: str) -> str:
    return design_id.replace("/", "_")


def _str(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _require(p: Path) -> str:
    if not p.is_file():
        raise FileNotFoundError(p)
    return str(p)


class _RefPicker:
    """Resolves the reference path for (design, component) under one reference mode."""

    def __init__(self, records_df: pd.DataFrame, root: Path, reference: str, rng: np.random.Generator):
        if reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
        self.root, self.reference, self.rng = root, reference, rng
        g = records_df[records_df.label == 0]
        self.genuine = {d: list(zip(s.image_id, s.component_id)) for d, s in g.groupby("design_id")}

    def __call__(self, design_id: str, component_id: str) -> str:
        tag, ref_dir = _tag(design_id), self.root / "references"
        if self.reference == "median":
            return _require(ref_dir / f"{tag}_fold{1 - fold_of(component_id)}.png")
        if self.reference == "specimen":          # registered to the fold median the query never entered
            return _require(ref_dir / f"{tag}_specimen_fold{1 - fold_of(component_id)}.png")
        pool = sorted(i for i, c in self.genuine.get(design_id, []) if c != component_id)
        if not pool:
            raise FileNotFoundError(f"no genuine canvas of {design_id} outside component {component_id}")
        return _require(self.root / "canvases" / f"{pool[self.rng.integers(len(pool))]}{CANVAS_EXT}")


def build_pairs(split_df: pd.DataFrame, records_df: pd.DataFrame, processed_root, reference="median",
                wrong_design_fraction=0.1, role="train", seed=0) -> list[dict]:
    """Samples of one split role as dicts of paths; wrong-design negatives appended when role == 'train'."""
    root = Path(processed_root)
    rng = np.random.default_rng(seed)
    pick = _RefPicker(records_df, root, reference, rng)
    rec = records_df.set_index("image_id")
    rows = split_df[split_df.role == role].sort_values("image_id")
    pairs = []
    for r in rows.itertuples(index=False):
        comp, label = str(r.component_id), int(r.label)
        valid = root / "valid" / f"{r.image_id}.png"
        pairs.append(dict(
            image_id=r.image_id, design_id=r.design_id, component_id=comp, label=label,
            path=_require(root / "canvases" / f"{r.image_id}{CANVAS_EXT}"),
            valid_path=str(valid) if valid.is_file() else None,
            mask_path=_require(root / "masks" / f"{r.image_id}.png") if label else None,
            zone_path=_require(root / "zones" / f"{_tag(r.design_id)}.png"),
            ref_path=pick(r.design_id, comp),
            subset=_str(r.subset), attack_family=_str(rec.at[r.image_id, "attack_family"]),
            wrong_design=False))
    if role == "train" and wrong_design_fraction > 0:
        pairs += _wrong_design(pairs, pick, wrong_design_fraction, rng)
    return pairs


def _wrong_design(pairs: list[dict], pick: _RefPicker, fraction: float, rng: np.random.Generator) -> list[dict]:
    """Copies of a fraction of the genuine samples paired with another design's reference, label 1, no mask."""
    genuine = [p for p in pairs if p["label"] == 0]
    designs = sorted({p["design_id"] for p in pairs})
    if len(designs) < 2 or not genuine:
        return []
    n = int(round(fraction * len(genuine)))
    out = []
    for i in sorted(rng.choice(len(genuine), size=n, replace=False)):
        p = dict(genuine[i])
        others = [d for d in designs if d != p["design_id"]]
        other = others[rng.integers(len(others))]
        p.update(ref_path=pick(other, p["component_id"]), label=1, mask_path=None, wrong_design=True)
        out.append(p)
    return out
