"""Checks for pairs.build_pairs and dataset.PairDataset on the processed Protocol A data.

Run: python tests/test_data_pairs.py (plain asserts, no pytest).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.data.dataset import PairDataset, make_loader  # noqa: E402
from refer_or_clear.data.pairs import build_pairs  # noqa: E402
from refer_or_clear.data.references import fold_of  # noqa: E402

P = ROOT / "data" / "processed"
split = pd.read_csv(P / "splits" / "protocol_a.csv")
records = pd.read_csv(P / "records.csv")


def check_pairs(pairs, role, reference, fraction):
    labels = {p["label"] for p in pairs}
    genuine = [p for p in pairs if not p["wrong_design"] and p["label"] == 0]
    wrong = [p for p in pairs if p["wrong_design"]]
    n_role = int((split.role == role).sum())
    assert len(pairs) == n_role + len(wrong), (len(pairs), n_role, len(wrong))
    assert labels == {0, 1}, labels
    if role == "train":
        assert len(wrong) == round(fraction * len(genuine)), (len(wrong), len(genuine))
        assert all(p["label"] == 1 and p["mask_path"] is None for p in wrong)
        assert all(not Path(p["ref_path"]).name.startswith(p["design_id"].replace("/", "_")) for p in wrong)
    else:
        assert not wrong
    for p in pairs:
        for k in ("path", "ref_path", "zone_path"):
            assert Path(p[k]).is_file(), p[k]
        assert (p["mask_path"] is None) == (p["label"] == 0 or p["wrong_design"]), p
        assert p["valid_path"] is None or Path(p["valid_path"]).is_file()
        assert all(k in p for k in ("subset", "attack_family", "component_id", "design_id"))
        if reference == "median":
            fold = int(re.search(r"_fold(\d)\.jpg$", p["ref_path"]).group(1))
            assert fold != fold_of(p["component_id"]), p
        elif reference == "specimen":
            assert "_specimen_fold" in p["ref_path"]
    if reference == "median":  # no component of the sample's fold is in the reference fold
        comps = records[records.label == 0].groupby("design_id").component_id.apply(set)
        for p in pairs:
            fold = int(re.search(r"_fold(\d)\.jpg$", p["ref_path"]).group(1))
            same = {c for c in comps[p["design_id"]] if fold_of(c) == fold}
            assert p["component_id"] not in same
    print(f"  {role:5s} {reference:8s}: {len(pairs):5d} pairs, {len(genuine)} genuine, "
          f"{sum(p['label'] for p in pairs)} altered, {len(wrong)} wrong-design")


def check_item(item, H, W):
    x, m, v = item["x"], item["mask"], item["valid"]
    assert x.shape == (8, H, W) and x.dtype == torch.float32, (x.shape, x.dtype)
    assert m.shape == (1, H, W) and v.shape == (1, H, W) and m.dtype == v.dtype == torch.float32
    for ch in (3, 7):
        assert set(x[ch].unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(x[3], x[7])
    assert set(m.unique().tolist()) <= {0.0, 1.0} and set(v.unique().tolist()) <= {0.0, 1.0}
    assert x[[0, 1, 2, 4, 5, 6]].min() >= -2.2 and x[[0, 1, 2, 4, 5, 6]].max() <= 2.7
    assert torch.isfinite(x).all()
    assert item["label"].dtype == torch.float32 and item["label"].item() in (0.0, 1.0)
    assert set(item["meta"]) == {"image_id", "design_id", "subset", "attack_family", "wrong_design"}
    if item["label"].item() == 0:
        assert m.sum() == 0
    if item["meta"]["wrong_design"]:
        assert torch.equal(m, v)


def main():
    FRACTION = 0.1
    print("pairs")
    for reference in ("median", "specimen"):
        for role in ("train", "val", "test"):
            pairs = build_pairs(split, records, P, reference=reference, wrong_design_fraction=FRACTION, role=role)
            check_pairs(pairs, role, reference, FRACTION)
    train = build_pairs(split, records, P, reference="median", role="train", seed=0)
    assert train == build_pairs(split, records, P, reference="median", role="train", seed=0), "not deterministic"
    single = build_pairs(split, records, P, reference="single", role="val")
    assert all("/canvases/" in p["ref_path"] for p in single)
    assert single == build_pairs(split, records, P, reference="single", role="val")
    test = build_pairs(split, records, P, reference="median", role="test")

    print("dataset")
    ds_train = PairDataset(train, train=True)
    ds_test = PairDataset(test, train=False)
    ds_eq = PairDataset(test, train=False, equalize_qf=75)
    wrong_idx = next(i for i, p in enumerate(train) if p["wrong_design"])
    altered_idx = next(i for i, p in enumerate(train) if p["label"] == 1 and not p["wrong_design"])
    for i in (0, altered_idx, wrong_idx):
        check_item(ds_train[i], 448, 704)
    for i in (0, 1, len(test) - 1):
        check_item(ds_test[i], 512, 800)
        check_item(ds_eq[i], 512, 800)
    assert ds_train[altered_idx]["mask"].sum() > 0
    assert not torch.equal(ds_test[0]["x"], ds_eq[0]["x"]), "equalize_qf changed nothing"
    assert torch.equal(ds_test[0]["x"], ds_test[0]["x"]), "test mode must be deterministic"
    assert not torch.equal(ds_train[0]["x"], ds_train[0]["x"]), "train mode must augment"
    batch = next(iter(make_loader(train[:4], True, batch_size=2, workers=0)))
    assert batch["x"].shape == (2, 8, 448, 704) and batch["label"].shape == (2,)
    assert batch["mask"].shape == batch["valid"].shape == (2, 1, 448, 704)
    assert len(batch["meta"]["image_id"]) == 2

    for name, ds in (("train", ds_train), ("test", ds_test)):
        t = time.perf_counter()
        for i in range(20):
            ds[i]
        print(f"  {name}: {(time.perf_counter() - t) / 20 * 1000:.1f} ms per item")
    print("ok")


if __name__ == "__main__":
    main()
