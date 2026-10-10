"""Protocol splits, grouped by provenance component.

Protocol A (anchor): the SIDTD paper's ten official template folds (kfold_split_templates), used as given,
one split per fold; each row carries ref_group 'it<k>' so the design medians of fold k are built from that
fold's training file only.

Protocol C: every SIDTD image (plus IDNet when staged) is the training pool (90/10 by component
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


REF_COL = "ref_group"
KFOLD_DIR = ("sidtd", "kfold_split_templates", "split_kfold")
N_FOLDS = 10


def _csv_ids(csv: Path) -> set[str]:
    p = pd.read_csv(csv)
    col = [c for c in p.columns if "path" in c.lower()][0]
    return {"sidtd_" + Path(x).stem for x in p[col]}


def protocol_a_kfold(records: pd.DataFrame, root: Path) -> dict[int, pd.DataFrame]:
    """Protocol A (anchor): fold k -> split frame, roles read from the paper's CSVs and never re-drawn."""
    d = Path(root).joinpath(*KFOLD_DIR)
    s0 = records[records.dataset == "sidtd"]
    known = set(s0.image_id)
    out = {}
    for k in range(N_FOLDS):
        ids = {role: _csv_ids(d / f"{role}_split_SIDTD_it_{k}.csv") for role in ("train", "val", "test")}
        for role, s in ids.items():
            if s - known:
                raise ValueError(f"fold {k} {role}: {len(s - known)} ids not in records, e.g. {sorted(s - known)[:3]}")
        if (ids["train"] & ids["val"]) or (ids["train"] & ids["test"]) or (ids["val"] & ids["test"]):
            raise ValueError(f"fold {k}: an image appears in two roles")
        s = s0.copy()
        s["protocol"], s["subset"], s[REF_COL] = "A", "", f"it{k}"
        s["role"] = ""
        for role, members in ids.items():
            s.loc[s.image_id.isin(members), "role"] = role
        if (s.role == "").any():
            raise ValueError(f"fold {k}: {int((s.role == '').sum())} SIDTD images are in no role")
        out[k] = s[COLS + [REF_COL]].reset_index(drop=True)
    return out


# The ten nationality groups hard-coded in the SIDTD paper's few-shot code
# (SIDTD/models/fsl_models/test_fsl.py, partitions_test); each names the four test nationalities.
FSL_GROUPS = ("svk_alb_aze_lva", "svk_rus_esp_aze", "grc_esp_fin_srb", "svk_alb_grc_est", "alb_grc_rus_lva",
              "alb_grc_est_rus", "svk_rus_lva_fin", "rus_aze_lva_fin", "svk_alb_lva_fin", "svk_alb_rus_fin")
NATIONALITY = {"alb": "sidtd/alb_id", "aze": "sidtd/aze_passport", "esp": "sidtd/esp_id", "est": "sidtd/est_id",
               "fin": "sidtd/fin_id", "grc": "sidtd/grc_passport", "lva": "sidtd/lva_passport",
               "rus": "sidtd/rus_internalpassport", "srb": "sidtd/srb_passport", "svk": "sidtd/svk_id"}


def protocol_b_groups(records: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Protocol B: group k -> split frame; the four nationalities of FSL_GROUPS[k] are the test set, the other six
    train, with a 10 percent validation set by provenance-component hash. References use the global fold medians."""
    s0 = records[records.dataset == "sidtd"]
    out = {}
    for k, name in enumerate(FSL_GROUPS):
        test_designs = {NATIONALITY[n] for n in name.split("_")}
        s = s0.copy()
        s["protocol"], s["subset"], s[REF_COL] = "B", "", ""
        is_test = s.design_id.isin(test_designs)
        s["role"] = "test"
        s.loc[~is_test, "role"] = _train_val(s[~is_test])
        if set(s[s.role == "test"].design_id) & set(s[s.role != "test"].design_id):
            raise ValueError(f"group {name}: a design is both test and train")
        out[k] = s[COLS + [REF_COL]].reset_index(drop=True)
    return out


def protocol_c(records: pd.DataFrame) -> pd.DataFrame:
    pool = records[records.dataset.isin(["sidtd", "idnet"])].copy()
    pool["protocol"], pool["subset"] = "C", ""
    pool["role"] = _train_val(pool)
    f = records[records.dataset == "fantasyid"].copy()
    f["protocol"], f["role"], f["subset"] = "C", "", ""
    is_test = f.path.str.startswith("fantasyid/test/")
    f.loc[is_test, "role"] = "test"
    f.loc[is_test & f.path.str.contains("/digital_3/"), "subset"] = "attack1"
    f.loc[is_test & ~f.path.str.contains("/digital_3/"), "subset"] = "heldout"
    f.loc[f.path.str.startswith("fantasyid/train/bonafide/"), "role"] = "reference_only"
    f = f[f.role != ""]
    return pd.concat([pool, f], ignore_index=True)[COLS]
