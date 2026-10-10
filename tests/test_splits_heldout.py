"""Protocol B nationality groups: python tests/test_splits_heldout.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear.data.splits import FSL_GROUPS, NATIONALITY, REF_COL, protocol_b_groups  # noqa: E402

RECORDS = pd.read_csv(ROOT / "data" / "processed" / "records.csv", keep_default_na=False)

PAPER = ("svk_alb_aze_lva", "svk_rus_esp_aze", "grc_esp_fin_srb", "svk_alb_grc_est", "alb_grc_rus_lva",
         "alb_grc_est_rus", "svk_rus_lva_fin", "rus_aze_lva_fin", "svk_alb_lva_fin", "svk_alb_rus_fin")


def test_groups_are_the_papers():
    assert FSL_GROUPS == PAPER


def test_each_group_holds_out_its_four_nationalities():
    groups = protocol_b_groups(RECORDS)
    assert sorted(groups) == list(range(10))
    tested = set()
    for k, df in groups.items():
        want = {NATIONALITY[n] for n in FSL_GROUPS[k].split("_")}
        assert set(df[df.role == "test"].design_id) == want, k
        assert not (set(df[df.role != "test"].design_id) & want), k
        assert set(df.role) == {"train", "val", "test"}, k
        assert (df.protocol == "B").all() and (df[REF_COL] == "").all()
        tr, va = set(df[df.role == "train"].component_id), set(df[df.role == "val"].component_id)
        assert not (tr & va), "a component is split across train and val"
        frac = (df.role == "val").sum() / (df.role != "test").sum()
        assert 0.05 < frac < 0.15, frac
        tested |= want
    assert len(tested) == 10, "every nationality is tested in at least one group"


if __name__ == "__main__":
    test_groups_are_the_papers()
    test_each_group_holds_out_its_four_nationalities()
    print("ok")
