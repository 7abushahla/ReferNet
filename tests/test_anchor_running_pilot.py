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


def test_running_pilot_has_separate_full_training_contract_without_unlocking_bounded_track():
    bounded = contract_for("D_running")
    full = contract_for("D_running_pilot")
    assert bounded.diagnostic_only
    assert not full.diagnostic_only
    assert full.architecture_id == "anchor_direct_running_ia18_pilot_v1"
    assert full.input_shape == bounded.input_shape
    assert full.bn_policy == bounded.bn_policy == "running"
    assert full.mask_weight == bounded.mask_weight == 0.0
    assert pilots.OFFICIAL_WEIGHT_SHA256["D_running_pilot"] == pilots.OFFICIAL_WEIGHT_SHA256["D_running"]
    assert pilots.evaluation_role("D_running", "smoke") == "train"
    assert pilots.evaluation_role("D_running_pilot", "smoke") == "val"


def test_running_full_recipe_changes_only_contract_identifier():
    bounded = resolve_pilot_config(ROOT / "configs/anchor_pilot_d_running.yaml")
    full = resolve_pilot_config(ROOT / "configs/anchor_pilot_d_running_pilot.yaml")
    assert {key for key in bounded if bounded[key] != full[key]} == {"track", "architecture_id"}


@pytest.fixture
def full_running_fixture(monkeypatch, tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic full pilot initialization")
    checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setitem(pilots.OFFICIAL_WEIGHT_SHA256, "D_running_pilot", checksum)
    opened = []

    def rows(records, split, role):
        assert role in ("train", "val"), "test pixels are forbidden"
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
        model.track = "D_running_pilot"
        model.architecture_id = "anchor_direct_running_ia18_pilot_v1"
        model.initialization_provenance["checkpoint_sha256"] = checksum
        return model

    monkeypatch.setattr(query, "load_role_rows", rows)
    monkeypatch.setattr(query, "AnchorQueryDataset", FixtureDataset)
    monkeypatch.setattr(pilots, "_model_builder", lambda track: builder)
    for name in ("records.csv", "split.csv"):
        (tmp_path / name).write_text("synthetic metadata authority")
    return dict(config_path=ROOT / "configs/anchor_pilot_d_running_pilot.yaml", raw_root=tmp_path,
                legacy_root=tmp_path, device="cpu", weights=weights, weights_sha256=checksum,
                records=tmp_path / "records.csv", split=tmp_path / "split.csv"), opened


def test_running_full_smoke_and_replay_use_validation_without_dense_targets(full_running_fixture, tmp_path):
    options, opened = full_running_fixture
    smoke = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="smoke")
    replay = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="replay")
    assert smoke["passed"] and replay["passed"]
    assert smoke["opened_roles"] == ["train", "val"]
    assert smoke["n_images"] == {"train": 32, "val": 32}
    assert smoke["evaluation_role"] == "val"
    assert smoke["inference_after"]["batch_sizes"] == [1, 2, 4, 8]
    assert opened == ["train", "val", "val"]
    summary = pilots._read_json(tmp_path / "smoke/val_summary.json")
    assert summary["role"] == "val" and summary["localization_metrics"] is None


def test_running_full_gate_does_not_accept_bounded_track_gate(full_running_fixture, tmp_path):
    options, opened = full_running_fixture
    for phase in ("overfit", "smoke"):
        assert pilots.run_pilot(**options, run_dir=tmp_path / phase, phase=phase)["passed"]
    replay = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="replay")
    replay["fresh_process"] = True
    pilots.write_json(tmp_path / "smoke/replay.json", replay)
    gate = pilots.check_pilot_gates(tmp_path, "pretrain")
    assert gate["passed"], gate["failures"]
    assert gate["tracks"] == {"overfit": "D_running_pilot", "smoke": "D_running_pilot"}
    identities = dict(gate["identities"], config_sha256=pilots.json_sha256(
        resolve_pilot_config(ROOT / "configs/anchor_pilot_d_running.yaml")))
    with pytest.raises(ValueError, match="matching identities"):
        pilots.validate_pretrain_initialization(gate, identities, gate["fresh_initialization"])
    assert opened == ["train", "train", "val", "val"]
