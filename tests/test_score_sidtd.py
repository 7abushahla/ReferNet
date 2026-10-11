"""Diagnostic scorer for the SIDTD paper's released template checkpoints: python tests/test_score_sidtd.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_sidtd_checkpoints as S  # noqa: E402

RAW = ROOT / "data" / "raw" / "sidtd"


def test_fold_rows_match_the_paper_file():
    rows = S.fold_rows(RAW, 4)
    assert len(rows) == 223, len(rows)
    assert all(p.is_file() for p, _ in rows[:20])
    assert {l for _, l in rows} == {0, 1}
    assert str(rows[0][0]).startswith(str(RAW / "templates" / "Images"))


def test_preprocess_matches_the_paper():
    img = np.zeros((600, 900, 3), np.uint8)
    img[..., 0] = 255   # a pure-blue BGR image as cv2 would read it
    x = S.preprocess(img, 299)
    assert x.shape == (3, 299, 299), x.shape
    # BGR -> RGB: the blue plane becomes channel 2, normalised with ImageNet mean/std
    assert abs(x[2, 0, 0] - (1.0 - 0.406) / 0.225) < 1e-5 and abs(x[0, 0, 0] - (0.0 - 0.485) / 0.229) < 1e-5


def test_checkpoint_for_fold_picks_the_fold_file():
    files = ["MIDV2020_resnet50_best_accuracy_n0.pth", "MIDV2020_resnet50_best_accuracy_n4.pth"]
    assert S.pick_checkpoint(files, 4).endswith("_n4.pth")
    try:
        S.pick_checkpoint(files, 7)
    except FileNotFoundError:
        return
    raise AssertionError("missing fold checkpoint not reported")


if __name__ == "__main__":
    test_fold_rows_match_the_paper_file()
    test_preprocess_matches_the_paper()
    test_checkpoint_for_fold_picks_the_fold_file()
    print("ok")
