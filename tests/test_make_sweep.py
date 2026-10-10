"""Sweep generator: python tests/test_make_sweep.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_sweep as MS  # noqa: E402


def run(*extra):
    d = tempfile.mkdtemp()
    MS.write_sweep(["--archs", "changerex_r18", "--references", "median", "--inits", "levir", "--seeds", "42",
                    "--name", "t", "--out", d, *extra])
    return {p.name: p.read_text() for p in Path(d).glob("*.sh") if p.name != "submit_all.sh"}


def test_folds_in_tags_and_flags():
    jobs = run("--protocols", "A", "--fusions", "single,difference", "--folds", "0-1")
    assert set(jobs) == {f"A-cex-{f}-median-levir-f{k}-s42.sh" for f in ("single", "difference") for k in (0, 1)}
    text = jobs["A-cex-single-median-levir-f1-s42.sh"]
    assert "--fold 1" in text and "configs/protocol_a.yaml" in text
    assert "--seed 42 \\\n        --fold 1 \\\n        --data-root" in text, "line continuation broken"
    assert 'RUN_DIR="results/runs/A-cex-single-median-levir-f1-s42"' in text
    assert "protocol_a.csv" not in text


def test_no_folds_keeps_old_tags():
    jobs = run("--protocols", "C", "--fusions", "difference")
    assert set(jobs) == {"C-cex-difference-median-levir-s42.sh"}
    assert "--fold" not in jobs["C-cex-difference-median-levir-s42.sh"]
    assert "configs/protocol_c.yaml" in jobs["C-cex-difference-median-levir-s42.sh"]


def test_protocol_d_and_fold_list():
    jobs = run("--protocols", "D", "--fusions", "difference", "--folds", "2,5")
    assert set(jobs) == {"D-cex-difference-median-levir-f2-s42.sh", "D-cex-difference-median-levir-f5-s42.sh"}
    assert "configs/protocol_d.yaml" in jobs["D-cex-difference-median-levir-f2-s42.sh"]


if __name__ == "__main__":
    test_folds_in_tags_and_flags()
    test_no_folds_keeps_old_tags()
    test_protocol_d_and_fold_list()
    print("ok")
