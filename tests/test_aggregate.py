"""Aggregator: python tests/test_aggregate.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_runs as AG  # noqa: E402


def write(runs: Path, tag: str, auc: float, acc: float, fold, designs=()):
    d = runs / tag
    d.mkdir(parents=True)
    block = {"auc": auc, "deepid_acc": acc}
    subsets = {f"design:{x}": {"native": {"auc": auc - 0.01, "deepid_acc": acc}} for x in designs}
    json.dump({"fold": fold, "val": {"auc": 0.99}, "timing": {"epochs": 40, "best_epoch": 30},
               "test": {"native": block, "equalized_qf75": block}, "subsets": subsets}, open(d / "metrics.json", "w"))


def make(control_auc=0.995, n=10):
    runs = Path(tempfile.mkdtemp())
    for k in range(n):
        write(runs, f"A-cex-single-median-levir-f{k}-s42", control_auc, 0.985, k)
        write(runs, f"A-cex-difference-median-levir-f{k}-s42", 0.999, 0.995, k)
    write(runs, "A-cex-single-median-levir-s42", 0.40, 0.40, None)   # superseded, no fold
    return runs


def test_superseded_runs_are_ignored():
    df = AG.load(make(), "A")
    assert set(df.fold.dropna().astype(int)) == set(range(10))
    assert not ((df.fusion == "single") & (df.auc < 0.5)).any()


def test_paper_table_has_published_rows_and_ours():
    t = AG.paper_table(AG.load(make(), "A"), "native")
    for name in ("EfficientNet-B3", "ResNet50", "ViT-L/16", "TransFG", "CoAARN", "control", "design median"):
        assert name in t, name
    assert "99.4 ± 1.0" in t and "98.5 ± 0.0" in t


def test_gate():
    ok, msg = AG.gate(AG.load(make(), "A"))
    assert ok, msg
    ok, msg = AG.gate(AG.load(make(control_auc=0.95), "A"))
    assert not ok and "AUC" in msg
    ok, msg = AG.gate(AG.load(make(n=9), "A"))
    assert not ok and "9 of 10" in msg


def test_gate_counts_distinct_folds():
    runs = Path(tempfile.mkdtemp())
    for k in range(5):
        for seed in (42, 43):
            write(runs, f"A-cex-single-median-levir-f{k}-s{seed}", 0.995, 0.985, k)
    ok, msg = AG.gate(AG.load(runs, "A"))
    assert not ok and "5 of 10" in msg, msg


def test_per_design_for_heldout():
    runs = Path(tempfile.mkdtemp())
    write(runs, "B-cex-difference-median-levir-f0-s42", 0.9, 0.9, 0, designs=("sidtd/alb_id", "sidtd/aze_passport"))
    write(runs, "B-cex-difference-median-levir-f1-s42", 0.8, 0.8, 1, designs=("sidtd/alb_id",))
    t = AG.per_design(AG.load(runs, "B"))
    assert "sidtd/alb_id" in t and "0.840" in t and "sidtd/aze_passport" in t


if __name__ == "__main__":
    test_superseded_runs_are_ignored()
    test_paper_table_has_published_rows_and_ours()
    test_gate()
    test_gate_counts_distinct_folds()
    test_per_design_for_heldout()
    print("ok")
