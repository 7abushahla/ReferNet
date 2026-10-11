#!/usr/bin/env python3
"""Verify matched fold-0 inputs or smoke results without constructing test pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear import evaluate as evaluation
from refer_or_clear.data.rectify import to_canvas
from diagnose_control import sha256

CONDITIONS = dict(direct="data/processed_direct_ablation", letterbox="data/processed_noreg",
                  registered="data/processed")


def verify_inputs(out):
    config = evaluation.load_config(ROOT / "configs/protocol_a.yaml")
    reference_rows = None
    report = {}
    for condition, directory in CONDITIONS.items():
        root = ROOT / directory
        split, records = evaluation.load_split(config, root, fold=0)
        rows = sorted((str(row.image_id), str(row.role), int(row.label)) for row in split.itertuples())
        if reference_rows is None:
            reference_rows = rows
        assert rows == reference_rows, f"fold IDs/roles/labels differ for {condition}"
        selected = {role: evaluation.role_pairs(split, records, root, role, "median", config, 42,
                                                wrong_design_fraction=0.) for role in ("train", "val")}
        assert {role: len(pairs) for role, pairs in selected.items()} == dict(train=1778, val=222)
        indexed = records.set_index("image_id")
        for role, pairs in selected.items():
            for pair in pairs:
                assert not pair["wrong_design"]
                image = cv2.imread(str(pair["path"]))
                assert image is not None and image.shape == (512, 800, 3)
                for key in ("zone_path", "valid_path", "mask_path"):
                    if pair[key]:
                        plane = cv2.imread(str(pair[key]), cv2.IMREAD_GRAYSCALE)
                        assert plane is not None and plane.shape == (512, 800), (condition, key, pair["image_id"])
                        if key == "mask_path" and pair["label"]:
                            assert (plane > 127).any(), (condition, pair["image_id"], "empty forgery mask")
                if condition != "registered":
                    source = cv2.imread(str(ROOT / "data/raw" / indexed.at[pair["image_id"], "path"]))
                    expected = to_canvas(source, mode="direct" if condition == "direct" else "letterbox")[0]
                    assert np.array_equal(image, expected), (condition, pair["image_id"], "pixel mismatch")
            print(condition, role, len(pairs), "verified", flush=True)
        report[condition] = dict(data_root=directory, split_sha256=sha256(root / "splits/protocol_a_it0.csv"),
                                  records_sha256=sha256(root / "records.csv"),
                                  transforms_sha256=sha256(root / "transforms.json"),
                                  counts={role: len(pairs) for role, pairs in selected.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(conditions=report, test_pairs_constructed=False), indent=2) + "\n")


def verify_smokes(run_roots, out):
    reference = None
    report = {}
    for directory in run_roots:
        root = Path(directory)
        info = json.loads((root / "train.json").read_text())
        metrics = json.loads((root / "metrics.json").read_text())
        reloaded = json.loads((root / "eval_val.json").read_text())
        config = info["config"]
        assert config["validation_only"] and config["strict_fp32"]
        assert config["bn_policy"] == "batch_stats" and config["eval_batch_size"] == 1
        assert info["precision"] == dict(matmul_tf32=False, cudnn_tf32=False, amp=False)
        assert set(info["n_pairs"]) == {"train", "val"}
        assert metrics["test"] == {} and metrics["subsets"] == {}
        assert np.isfinite(metrics["val"]["auc"])
        assert reloaded["split"] == "val"
        assert reloaded["precision"] == info["precision"]
        assert abs(reloaded["val"]["auc"] - metrics["val"]["auc"]) < 1e-7
        comparable = {name: value for name, value in config.items() if name not in ("tag", "data_root")}
        if reference is None:
            reference = comparable
        assert comparable == reference, "smoke training configurations differ beyond input directory/tag"
        report[root.name] = dict(val_auc=metrics["val"]["auc"], config=config, n_params=info["n_params"])
    assert len(report) == 3 and len({row["n_params"] for row in report.values()}) == 1
    assert len({row["config"]["data_root"] for row in report.values()}) == 3, "input conditions reuse the same data root"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(passed=True, runs=report), indent=2) + "\n")
    print("all three validation-only smoke runs verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smokes", nargs=3)
    args = parser.parse_args()
    if args.smokes:
        verify_smokes(args.smokes, Path(args.out))
    else:
        verify_inputs(Path(args.out))
