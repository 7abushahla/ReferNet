"""train.resolve, the selection rule and fold splits: python tests/test_train_config.py"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear import evaluate as E  # noqa: E402
from refer_or_clear import train as T  # noqa: E402

CFG_A = E.load_config(ROOT / "configs" / "protocol_a.yaml")


def rc_for(fusion, fold=3):
    a = T.parse_args(["--config", "configs/protocol_a.yaml", "--arch", "changerex_r18", "--fusion", fusion,
                      "--init", "levir", "--run-dir", "/tmp/x", "--fold", str(fold)])
    return T.resolve(a, CFG_A)


def test_control_gets_no_wrong_design_pairs():
    assert rc_for("single")["wrong_design_fraction"] == 0.0
    assert rc_for("attention_single")["wrong_design_fraction"] == 0.0
    assert rc_for("single_noex")["wrong_design_fraction"] == 0.0
    assert rc_for("difference")["wrong_design_fraction"] == 0.1


def test_fold_select_and_jpeg():
    rc = rc_for("difference", fold=7)
    assert rc["fold"] == 7 and rc["select"] == "accuracy"
    assert T.jpeg_range(rc) is None
    assert T.jpeg_range({"augment": {"jpeg_quality": [60, 100]}}) == (60, 100)


def test_batch_stats_uses_one_image_per_validation_and_test_batch():
    args = T.parse_args(["--config", "configs/protocol_a.yaml", "--arch", "changerex_r18",
                         "--fusion", "single_noex", "--init", "levir", "--run-dir", "/tmp/x",
                         "--fold", "0", "--bn-policy", "batch_stats"])
    resolved = T.resolve(args, CFG_A)
    assert resolved["batch_size"] == 8
    assert resolved["eval_batch_size"] == 1
    assert resolved["bn_policy"] == "batch_stats"
    assert rc_for("single_noex")["bn_policy"] == "running"


def test_selection_key():
    k = T.selection_key
    assert k("accuracy", 0.99, 0.90, 0.1, 0.1) > k("accuracy", 0.98, 1.00, 0.1, 0.1)   # accuracy first
    assert k("accuracy", 0.99, 0.95, 0.1, 0.1) > k("accuracy", 0.99, 0.94, 0.1, 0.1)   # AUC breaks ties
    assert not (k("accuracy", 1.0, 1.0, 0.1, 0.1) > k("accuracy", 1.0, 1.0, 0.1, 0.1))  # full tie keeps the earlier
    assert k("auc", 0.5, 0.9, 0.1, 0.1) > k("auc", 0.9, 0.8, 0.1, 0.1)
    assert k("accuracy", None, None, 0.2, 0.1) > k("accuracy", None, None, 0.3, 0.1)
    assert k("accuracy", 0.9, None, 0.2, 0.1) == (0.9, -math.inf)


def test_validation_only_and_strict_precision():
    import torch
    args = T.parse_args(["--config", "configs/protocol_a.yaml", "--arch", "changerex_r18",
                         "--fusion", "single_noex", "--init", "levir", "--run-dir", "/tmp/x",
                         "--validation-only", "--strict-fp32"])
    resolved = T.resolve(args, CFG_A)
    assert resolved["validation_only"] and resolved["strict_fp32"]
    assert T.evaluation_roles(resolved) == ("train", "val")
    assert T.evaluation_roles(rc_for("single_noex")) == ("train", "val", "test")
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        E.configure_precision(resolved)
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original
    args.amp = True
    try:
        T.resolve(args, CFG_A)
    except ValueError:
        pass
    else:
        raise AssertionError("strict FP32 accepted mixed precision")


def test_load_split_formats_the_fold():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "splits").mkdir()
        pd.DataFrame([dict(image_id="i", protocol="A", role="test", subset="", component_id="c",
                           design_id="x/d", label=0, ref_group="it3")]).to_csv(d / "splits" / "protocol_a_it3.csv", index=False)
        pd.DataFrame([dict(image_id="i")]).to_csv(d / "records.csv", index=False)
        split, _ = E.load_split(CFG_A, d, fold=3)
        assert list(split.ref_group) == ["it3"]
        try:
            E.load_split(CFG_A, d)
        except ValueError as e:
            assert "fold" in str(e)
        else:
            raise AssertionError("a {fold} split without --fold was accepted")


def test_explicit_localization_weight_override():
    args = T.parse_args(["--config", "configs/protocol_a.yaml", "--arch", "changerex_r18",
                         "--fusion", "single_noex", "--init", "levir", "--run-dir", "/tmp/x",
                         "--mask-loss-weight", "0"])
    assert T.resolve(args, CFG_A)["mask_weight"] == 0.
    assert rc_for("single_noex")["mask_weight"] == 3.
    for invalid in (-1., float("nan"), float("inf")):
        args.mask_loss_weight = invalid
        try:
            T.resolve(args, CFG_A)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mask weight accepted")


def test_role_pairs_override_reaches_build_pairs():
    seen = {}

    def fake(split_df, records_df, root, reference, wrong_design_fraction, role, seed):
        seen["frac"] = wrong_design_fraction
        return []
    import refer_or_clear.data.pairs as P
    orig, P.build_pairs = P.build_pairs, fake
    try:
        split = pd.DataFrame([dict(role="train")])
        E.role_pairs(split, None, "x", "train", "median", CFG_A, 42, wrong_design_fraction=0.0)
        assert seen["frac"] == 0.0
        E.role_pairs(split, None, "x", "train", "median", CFG_A, 42)
        assert seen["frac"] == 0.1
    finally:
        P.build_pairs = orig


if __name__ == "__main__":
    test_control_gets_no_wrong_design_pairs()
    test_fold_select_and_jpeg()
    test_batch_stats_uses_one_image_per_validation_and_test_batch()
    test_selection_key()
    test_validation_only_and_strict_precision()
    test_load_split_formats_the_fold()
    test_explicit_localization_weight_override()
    test_role_pairs_override_reaches_build_pairs()
    print("ok")
