"""Protocol splits, grouped by provenance component.

Protocol A (SIDTD official few-shot split): the published train nationalities are the training pool
(90/10 train/val by component hash), the published test nationalities are the test set; membership
is read from the release CSVs so it is the authors', not ours.

Protocol B: every SIDTD image (plus IDNet when staged) is the training pool (90/10 by component
hash); the FantasyID test release is the test set with a subset tag: 'heldout' = the 300 bona fide
plus the face and text attacks on the three held-out designs (headline); 'attack1' = the 786 text
rewrites on the ten train-val designs. FantasyID train bona fide images carry role 'reference_only':
they build the medians for the ten train-val designs and are never training or test samples.
FantasyID train attacks are excluded entirely.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

VAL_FRACTION = 0.1
COLS = ["image_id", "protocol", "role", "subset", "component_id", "design_id", "label"]


def _hash01(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _train_val(df: pd.DataFrame) -> pd.Series:
    return df.component_id.map(lambda c: "val" if _hash01("val:" + c) < VAL_FRACTION else "train")


def _sidtd_official_fewshot(root: Path):
    d = root / "sidtd" / "split_shot"

    def ids(csv):
        p = pd.read_csv(csv)
        col = [c for c in p.columns if "path" in c.lower()][0]
        return {"sidtd_" + Path(x).stem for x in p[col]}
    return ids(d / "train_split_SIDTD.csv"), ids(d / "test_split_SIDTD.csv")


def protocol_a(records: pd.DataFrame, root: Path) -> pd.DataFrame:
    tr, te = _sidtd_official_fewshot(root)
    s = records[records.dataset == "sidtd"].copy()
    s["protocol"], s["subset"], s["role"] = "A", "", ""
    s.loc[s.image_id.isin(te), "role"] = "test"
    pool = s.image_id.isin(tr)
    s.loc[pool, "role"] = _train_val(s[pool])
    s = s[s.role != ""]
    assert not (set(s[s.role == "test"].design_id) & set(s[s.role != "test"].design_id)), "designs leak across A"
    return s[COLS]


def protocol_b(records: pd.DataFrame) -> pd.DataFrame:
    pool = records[records.dataset.isin(["sidtd", "idnet"])].copy()
    pool["protocol"], pool["subset"] = "B", ""
    pool["role"] = _train_val(pool)
    f = records[records.dataset == "fantasyid"].copy()
    f["protocol"], f["role"], f["subset"] = "B", "", ""
    is_test = f.path.str.startswith("fantasyid/test/")
    f.loc[is_test, "role"] = "test"
    f.loc[is_test & f.path.str.contains("/digital_3/"), "subset"] = "attack1"
    f.loc[is_test & ~f.path.str.contains("/digital_3/"), "subset"] = "heldout"
    f.loc[f.path.str.startswith("fantasyid/train/bonafide/"), "role"] = "reference_only"
    f = f[f.role != ""]
    return pd.concat([pool, f], ignore_index=True)[COLS]
