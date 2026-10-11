from importlib import import_module, util
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def resolver():
    assert util.find_spec("refer_or_clear.pilot_config") is not None, "pilot config interface missing"
    return import_module("refer_or_clear.pilot_config").resolve_pilot_config


def write_config(tmp_path, value):
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.mark.parametrize("track,architecture,batch,eval_batch,mask,bn", [
    ("R", "anchor_rgb_r50_v1", 32, 32, 0.0, "running"),
    ("D", "anchor_direct_ia18_v1", 8, 1, 0.0, "batch_stats"),
    ("N", "anchor_absent_ia18_v1", 8, 8, 3.0, "running"),
])
def test_resolved_track_defaults(resolver, tmp_path, track, architecture, batch, eval_batch, mask, bn):
    config = resolver(write_config(tmp_path, {"track": track}))
    assert config["schema_version"] == "anchor_pilot_v1"
    assert config["architecture_id"] == architecture
    assert (config["batch_size"], config["eval_batch_size"], config["mask_weight"], config["bn_policy"]) == (batch, eval_batch, mask, bn)
    assert (config["seed"], config["fold"], config["epochs"]) == (42, 0, 100)
    assert config["records_csv"] == "data/processed/records.csv"
    assert config["split_csv"] == "data/processed/splits/protocol_a_it0.csv"
    assert config["precision"] == dict(dtype="float32", amp=False, cudnn_tf32=False, matmul_tf32=False)
    assert config["selection"] == dict(primary="accuracy", tie_break="auc", exact_tie="earlier_epoch")
    assert config["normalization"] == dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    assert config["drop_last"] is (track != "R") and config["eval_drop_last"] is False
    assert config["accumulation_steps"] == (2 if track == "R" else 1)
    assert "weights_path" not in config and "weights_sha256" not in config


@pytest.mark.parametrize("track", ["R", "D", "N"])
def test_checked_in_recipes_resolve_to_exact_defaults(resolver, tmp_path, track):
    assert resolver(ROOT / "configs" / f"anchor_pilot_{track.lower()}.yaml") == resolver(write_config(tmp_path, {"track": track}))


def test_rgb_scientific_recipe(resolver, tmp_path):
    config = resolver(write_config(tmp_path, {"track": "R"}))
    assert config["optimizer"] == dict(name="sgd", lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False)
    assert config["scheduler"] == dict(name="reduce_on_plateau", monitor="val_ce", mode="min", factor=0.9,
                                      patience=1, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-6)
    augment = config["augment"]
    assert augment["preset"] == "raw_rgb_299_v1"
    assert augment["crop_scale"] == [0.8, 1.0] and augment["crop_ratio"] == [0.75, 4 / 3]
    assert augment["crop_attempts"] == 10 and augment["crop_ratio_sampling"] == "log_uniform"
    assert augment["brightness_offset"] == [-0.2, 0.2]
    assert augment["contrast_factor"] == [0.8, 1.2]
    assert augment["brightness_units"] == "255" and augment["clip"] == "uint8_0_255"


@pytest.mark.parametrize("track", ["D", "N"])
def test_legacy_scientific_recipe(resolver, tmp_path, track):
    config = resolver(write_config(tmp_path, {"track": track}))
    assert config["data_contract"] == "legacy_geometry_diagnostic"
    assert config["optimizer"] == dict(name="adamw", lr_backbone=1e-4, lr_new=1e-3, weight_decay=0.05,
                                      betas=[0.9, 0.999], eps=1e-8)
    assert config["scheduler"] == dict(name="warmup_cosine", warmup_iters=1000, warmup_cap_fraction=0.1)
    assert config["augment"]["crop"] is None and config["augment"]["jpeg_quality"] is None
    assert config["augment"]["affine_px"] == 8
    assert config["augment"]["jitter_px"] == 2 and config["augment"]["jitter_p"] == 0.25
    assert config["augment"]["photometric"] is True
    assert config["comparison"] == ("non_stream_matched" if track == "D" else "candidate_package")


@pytest.mark.parametrize("patch", [
    {"track": "unknown"}, {"architecture_id": "historical_model"}, {"data_contract": "legacy_geometry_diagnostic"},
    {"schema_version": "old"}, {"bn_policy": "batch_stats"}, {"mask_weight": 3},
    {"unknown": 3}, {"weights_path": "offline.pt"}, {"optimizer": {"lr_typo": 0.01}},
    {"optimizer": {"name": "adamw"}}, {"precision": {"amp": True}}, {"precision": {"matmul_tf32": True}},
    {"augment": {"jpeg_quality": [60, 100]}}, {"augment": {"crop_scale": [1.0, 0.8]}},
    {"epochs": 0}, {"batch_size": True}, {"seed": -1}, {"fold": 10},
    {"epochs": 2.5}, {"epochs": "100"}, {"selection": {"primary": "auc"}},
    {"optimizer": {"lr": float("nan")}}, {"records_csv": ""}, {"augment": None},
])
def test_invalid_or_unknown_settings_fail_closed(resolver, tmp_path, patch):
    with pytest.raises(ValueError):
        resolver(write_config(tmp_path, {"track": "R", **patch}))


def test_duplicate_yaml_keys_fail_closed(resolver, tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("track: R\ntrack: D\n")
    with pytest.raises(ValueError, match="duplicate"):
        resolver(path)


@pytest.mark.parametrize("value", [None, [], "R", {"seed": 42}])
def test_config_requires_mapping_and_explicit_track(resolver, tmp_path, value):
    with pytest.raises(ValueError):
        resolver(write_config(tmp_path, value))


def test_defaults_are_independent_and_runtime_metadata_paths_remain_relative(resolver, tmp_path):
    path = write_config(tmp_path, {"track": "R", "records_csv": "custom/records.csv", "split_csv": "custom/fold.csv"})
    first = resolver(path)
    first["augment"]["crop_scale"][0] = 0.1
    second = resolver(path)
    assert second["augment"]["crop_scale"] == [0.8, 1.0]
    assert second["records_csv"] == "custom/records.csv" and second["split_csv"] == "custom/fold.csv"


def test_fold_default_split_path_is_resolved_from_fold(resolver, tmp_path):
    config = resolver(write_config(tmp_path, {"track": "D", "fold": 1}))
    assert config["split_csv"] == "data/processed/splits/protocol_a_it1.csv"


def test_duplicate_nested_yaml_keys_fail_closed(resolver, tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("track: R\nprecision:\n  amp: false\n  amp: true\n")
    with pytest.raises(ValueError, match="duplicate"):
        resolver(path)
