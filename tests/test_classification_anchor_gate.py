"""Classification anchor must match the joint-loss control except for supervision."""
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def fixture(root, seed=42, map_loss=0.):
    config = dict(tag="joint", data_root="original", mask_weight=3., seed=42, epochs=100,
                  limit=None, validation_only=True, strict_fp32=True, bn_policy="batch_stats", eval_batch_size=1)
    baseline = root / "baseline.json"
    baseline.write_text(json.dumps(dict(config=config, n_params=123)))
    run = root / "classification"
    run.mkdir()
    candidate = dict(config, tag="classification", data_root="scratch", mask_weight=0., seed=seed, epochs=1, limit=32)
    precision = dict(matmul_tf32=False, cudnn_tf32=False, amp=False)
    (run / "train.json").write_text(json.dumps(dict(config=candidate, n_params=123,
        n_pairs=dict(train=32, val=32), precision=precision,
        supervision=dict(image_labels=True, localization_targets=False))))
    (run / "metrics.json").write_text(json.dumps(dict(val=dict(auc=.75), test={}, subsets={})))
    (run / "eval_val.json").write_text(json.dumps(dict(split="val", val=dict(auc=.75), precision=precision)))
    with (run / "log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "bce_map", "dice_map"))
        writer.writeheader()
        writer.writerow(dict(epoch=1, bce_map=map_loss, dice_map=0.))
    return run, baseline


def test_gate_rejects_changed_recipe_or_dense_loss():
    assert (ROOT / "scripts/check_classification_anchor.py").exists(), "anchor gate missing"
    from check_classification_anchor import verify_run
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run, baseline = fixture(root)
        verify_run(run, baseline, root / "accepted.json", smoke=True)
        assert json.loads((root / "accepted.json").read_text())["passed"]
    for seed, map_loss in ((43, 0.), (42, .01)):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, baseline = fixture(root, seed=seed, map_loss=map_loss)
            try:
                verify_run(run, baseline, root / "accepted.json", smoke=True)
            except AssertionError:
                pass
            else:
                raise AssertionError("unmatched classification anchor accepted")
            assert not (root / "accepted.json").exists()


if __name__ == "__main__":
    test_gate_rejects_changed_recipe_or_dense_loss()
    print("ok: classification anchor gate")
