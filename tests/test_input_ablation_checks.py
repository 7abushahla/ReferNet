"""Smoke acceptance rejects test evaluation and duplicated input conditions."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_input_ablation import verify_smokes


def fixtures(root, duplicate=False, test_used=False):
    runs = []
    for condition in ("direct", "letterbox", "registered"):
        directory = root / condition
        directory.mkdir()
        config = dict(validation_only=True, strict_fp32=True, bn_policy="batch_stats",
                      eval_batch_size=1, tag=condition, data_root="same" if duplicate else condition)
        info = dict(config=config, precision=dict(matmul_tf32=False, cudnn_tf32=False, amp=False),
                    n_pairs=dict(train=32, val=32), n_params=123)
        metrics = dict(test={"native": {}} if test_used else {}, subsets={}, val=dict(auc=.75))
        (directory / "train.json").write_text(json.dumps(info))
        (directory / "metrics.json").write_text(json.dumps(metrics))
        (directory / "eval_val.json").write_text(json.dumps(dict(
            split="val", precision=info["precision"], val=dict(auc=.75))))
        runs.append(directory)
    return runs


def test_smoke_acceptance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verify_smokes(fixtures(root), root / "accepted.json")
        assert json.loads((root / "accepted.json").read_text())["passed"]
    for duplicate, test_used in ((True, False), (False, True)):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                verify_smokes(fixtures(root, duplicate, test_used), root / "accepted.json")
            except AssertionError:
                pass
            else:
                raise AssertionError("invalid smoke accepted")
            assert not (root / "accepted.json").exists()


if __name__ == "__main__":
    test_smoke_acceptance()
    print("ok: smoke acceptance checks")
