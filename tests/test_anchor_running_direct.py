import hashlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear import pilots
from refer_or_clear.anchor_contract import contract_for
from refer_or_clear.data import query
from refer_or_clear.pilot_config import resolve_pilot_config
from test_anchor_training import FixtureDataset, TinyPilot, fixture_rows


def test_running_direct_changes_only_track_architecture_and_normalization():
    original = resolve_pilot_config(ROOT / "configs/anchor_pilot_d.yaml")
    variant = resolve_pilot_config(ROOT / "configs/anchor_pilot_d_running.yaml")
    assert {key for key in original if original[key] != variant[key]} == {"track", "architecture_id", "bn_policy"}
    assert variant["architecture_id"] == "anchor_direct_running_ia18_v1"
    assert variant["bn_policy"] == "running"
    assert contract_for("D_running").diagnostic_only
    assert not contract_for("D").diagnostic_only


def test_running_direct_full_training_stops_before_weight_or_data_access(tmp_path):
    with pytest.raises(ValueError, match="diagnostic.only"):
        pilots.run_pilot(ROOT / "configs/anchor_pilot_d_running.yaml", tmp_path, tmp_path, tmp_path / "run",
                         "cpu", "train", weights=tmp_path / "missing.pth", weights_sha256="a" * 64)
    assert not (tmp_path / "run").exists()


@pytest.fixture
def running_direct_fixture(monkeypatch, tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic bounded direct initialization")
    checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setitem(pilots.OFFICIAL_WEIGHT_SHA256, "D_running", checksum)
    opened = []

    def rows(records, split, role):
        assert role == "train", "bounded diagnostic must not access validation or test pixels"
        opened.append(role)
        return fixture_rows(role)

    def builder(**kwargs):
        model = TinyPilot("D")
        with torch.no_grad():
            model.encoder.weight.zero_()
            model.encoder.bias.zero_()
            model.encoder.weight[0, 0] = 4.0
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.weight[0, 0] = 4.0
        model.track = "D_running"
        model.architecture_id = "anchor_direct_running_ia18_v1"
        model.initialization_provenance["checkpoint_sha256"] = checksum
        return model

    monkeypatch.setattr(query, "load_role_rows", rows)
    monkeypatch.setattr(query, "AnchorQueryDataset", FixtureDataset)
    monkeypatch.setattr(pilots, "_model_builder", lambda track: builder)
    for name in ("records.csv", "split.csv"):
        (tmp_path / name).write_text("synthetic metadata authority")
    options = dict(config_path=ROOT / "configs/anchor_pilot_d_running.yaml", raw_root=tmp_path,
                   legacy_root=tmp_path, device="cpu", weights=weights,
                   weights_sha256=checksum, records=tmp_path / "records.csv", split=tmp_path / "split.csv")
    return options, opened


def test_running_direct_smoke_and_replay_open_training_only(running_direct_fixture, tmp_path):
    options, opened = running_direct_fixture
    smoke = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="smoke")
    replay = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="replay")
    assert smoke["passed"] and replay["passed"]
    assert smoke["opened_roles"] == ["train"]
    assert smoke["n_images"] == {"train": 32}
    assert smoke["evaluation_role"] == "train"
    assert smoke["inference_after"]["batch_sizes"] == [1, 2, 4, 8]
    assert smoke["inference_after"]["companion_invariance"]
    assert opened == ["train", "train"]
    assert torch.isfinite(torch.tensor(replay["summary"]["image_loss"]))


@pytest.mark.parametrize("violation", [None, "validation_access", "validation_evaluation", "bad_invariance"])
def test_running_direct_pretrain_gate_rejects_scope_and_invariance_violations(running_direct_fixture, tmp_path, violation):
    options, opened = running_direct_fixture
    for phase in ("overfit", "smoke"):
        outcome = pilots.run_pilot(**options, run_dir=tmp_path / phase, phase=phase)
        assert outcome["passed"]
    replay = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="replay")
    replay["fresh_process"] = True
    pilots.write_json(tmp_path / "smoke/replay.json", replay)
    info_path = tmp_path / "smoke/train.json"
    info = pilots._read_json(info_path)
    if violation == "validation_access":
        info["opened_roles"] = ["train", "val"]
    elif violation == "validation_evaluation":
        info["evaluation_role"] = "val"
    elif violation == "bad_invariance":
        info["inference_after"]["passed"] = False
    pilots.write_json(info_path, info)
    gate = pilots.check_pilot_gates(tmp_path, "pretrain")
    assert gate["passed"] == (violation is None), gate["failures"]
    assert opened == ["train", "train", "train"]
