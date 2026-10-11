#!/usr/bin/env python3
"""Verify a classification-only run against the registered joint-loss control."""
import argparse
import csv
import json
import math
from pathlib import Path


def verify_run(run, baseline, output, smoke):
    run, baseline, output = Path(run), Path(baseline), Path(output)
    reference = json.loads(baseline.read_text())
    info = json.loads((run / "train.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    replay = json.loads((run / "eval_val.json").read_text())
    config = info["config"]
    assert config["mask_weight"] == 0. and reference["config"]["mask_weight"] == 3.
    ignored = {"tag", "data_root", "epochs", "limit", "mask_weight"}
    assert {key: value for key, value in config.items() if key not in ignored} == {
        key: value for key, value in reference["config"].items() if key not in ignored}, "training recipe changed"
    assert config["validation_only"] and config["strict_fp32"]
    assert config["eval_batch_size"] == 1
    assert config["epochs"] == (1 if smoke else 100)
    assert config["limit"] == (32 if smoke else None)
    assert info["n_pairs"] == (dict(train=32, val=32) if smoke else dict(train=1778, val=222))
    assert info["n_params"] == reference["n_params"]
    assert info["supervision"] == dict(image_labels=True, localization_targets=False)
    assert info["precision"] == dict(matmul_tf32=False, cudnn_tf32=False, amp=False)
    assert metrics["test"] == {} and metrics["subsets"] == {}
    assert replay["split"] == "val" and replay["precision"] == info["precision"]
    assert math.isfinite(metrics["val"]["auc"])
    assert abs(metrics["val"]["auc"] - replay["val"]["auc"]) < 1e-7
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert len(rows) == config["epochs"]
    assert all(float(row["bce_map"]) == 0. and float(row["dice_map"]) == 0. for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(passed=True, run=str(run), reference=str(baseline),
                                     config=config, val_auc=metrics["val"]["auc"],
                                     localization_supervision=False), indent=2) + "\n")
    print("classification anchor configuration, zero dense loss and validation replay verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    verify_run(arguments.run, arguments.baseline, arguments.out, arguments.smoke)
