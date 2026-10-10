"""Protocol A folds and per-fold references: python tests/test_splits_anchor.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear.data import pairs as P  # noqa: E402
from refer_or_clear.data.references import build_references, fold_of  # noqa: E402
from refer_or_clear.data.splits import REF_COL, protocol_a_kfold  # noqa: E402

RAW = ROOT / "data" / "raw"
RECORDS = pd.read_csv(ROOT / "data" / "processed" / "records.csv", keep_default_na=False)
KDIR = RAW / "sidtd" / "kfold_split_templates" / "split_kfold"


def test_folds_match_the_paper_files():
    folds = protocol_a_kfold(RECORDS, RAW)
    assert sorted(folds) == list(range(10))
    n_sidtd = int((RECORDS.dataset == "sidtd").sum())
    tests = []
    for k, df in folds.items():
        assert len(df) == n_sidtd, (k, len(df))
        assert set(df[REF_COL]) == {f"it{k}"}
        for role in ("train", "val", "test"):
            n_csv = len(pd.read_csv(KDIR / f"{role}_split_SIDTD_it_{k}.csv"))
            assert int((df.role == role).sum()) == n_csv, (k, role)
        tests += list(df[df.role == "test"].image_id)
    assert len(tests) == len(set(tests)) == n_sidtd, "every image is tested exactly once"
    assert (folds[4].role == "train").sum() == 1776 and (folds[9].role == "test").sum() == 223


def _toy_canvases(d: Path, comps: list[str]) -> pd.DataFrame:
    (d / "canvases").mkdir()
    rows = []
    for i, c in enumerate(comps):
        cv2.imwrite(str(d / "canvases" / f"g{i}.png"), np.full((8, 8, 3), 10 * (i + 1), np.uint8))
        rows.append(dict(image_id=f"g{i}", design_id="x/d", identity_id=c, component_id=c, label=0))
    return pd.DataFrame(rows)


def test_group_references_use_only_the_given_records():
    comps = [f"c{i}" for i in range(12)]
    assert {fold_of(c) for c in comps} == {0, 1}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = _toy_canvases(d, comps)
        sub = rec[rec.component_id.isin(comps[:8])]
        build_references(sub, d / "canvases", d / "references", group="it3", write_all=False)
        names = sorted(p.name for p in (d / "references").glob("*.png"))
        assert "x_d_it3_fold0.png" in names and "x_d_it3_fold1.png" in names, names
        assert not any("_all" in n for n in names), names
        half = [i for i, c in enumerate(comps[:8]) if fold_of(c) == 0]
        expected = np.median([10 * (i + 1) for i in half])
        got = cv2.imread(str(d / "references" / "x_d_it3_fold0.png"))[0, 0, 0]
        assert abs(int(got) - int(expected)) <= 1, (got, expected)


def test_group_reference_fails_loudly_on_an_empty_half():
    comps = [c for c in (f"c{i}" for i in range(40)) if fold_of(c) == 0][:3]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = _toy_canvases(d, comps)
        try:
            build_references(rec, d / "canvases", d / "references", group="it0", write_all=False)
        except ValueError as e:
            assert "x/d" in str(e) and "it0" in str(e)
            return
        raise AssertionError("an empty half produced no error")


def test_picker_resolves_group_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "references").mkdir()
        for name in ("x_d_it2_fold0.png", "x_d_it2_fold1.png", "x_d_fold0.png", "x_d_fold1.png"):
            (d / "references" / name).write_bytes(b"0")
        rec = pd.DataFrame([dict(image_id="g", design_id="x/d", component_id="c1", label=0)])
        pick = P._RefPicker(rec, d, "median", np.random.default_rng(0))
        other = 1 - fold_of("c1")
        assert pick("x/d", "c1", "it2").endswith(f"x_d_it2_fold{other}.png")
        assert pick("x/d", "c1").endswith(f"x_d_fold{other}.png")


if __name__ == "__main__":
    test_folds_match_the_paper_files()
    test_group_references_use_only_the_given_records()
    test_group_reference_fails_loudly_on_an_empty_half()
    test_picker_resolves_group_files()
    print("ok")
