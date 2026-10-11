import copy
import hashlib
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import cv2
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.data.query import AnchorQueryDataset as RealAnchorQueryDataset


@pytest.fixture
def pilots():
    if importlib.util.find_spec("refer_or_clear.pilots") is not None:
        return importlib.import_module("refer_or_clear.pilots")
    class Runtime:
        def __getattr__(self, name):
            assert importlib.util.find_spec("refer_or_clear.pilots") is not None, "pilot runtime is not implemented"
            return getattr(importlib.import_module("refer_or_clear.pilots"), name)
    return Runtime()


class TinyPilot(nn.Module):
    def __init__(self, track="R"):
        super().__init__()
        self.track = track
        self.architecture_id = {"R": "anchor_rgb_r50_v1", "D": "anchor_direct_ia18_v1", "N": "anchor_absent_ia18_v1"}[track]
        self.encoder = nn.Linear(2, 3)
        self.head = nn.Linear(3, 2 if track == "R" else 1)
        self.initialization_provenance = {
            "requested_initialization": "IMAGENET1K_V1" if track == "R" else "levir",
            "checkpoint_path": "offline.pth", "checkpoint_sha256": "a" * 64,
            "loaded_keys": ["encoder.weight", "encoder.bias"], "missing_keys": [], "unexpected_keys": [],
            "pretrained_tensor_sha256": "b" * 64,
        }

    def forward(self, query):
        logits = self.head(self.encoder(query))
        if self.track == "R":
            return {"class_logits": logits, "attack_logit": logits[:, 1] - logits[:, 0]}
        outputs = {"attack_logit": logits[:, 0]}
        if self.track == "N":
            outputs["map_logit"] = logits[:, :1, None, None].expand(-1, 1, 2, 2)
        return outputs

    def parameter_groups(self):
        return list(self.encoder.parameters()), list(self.head.parameters())


def samples(count=5, masks=False):
    rows = []
    for index in range(count):
        row = {"query": torch.tensor([index / 5, 1.0]), "label": index % 2,
               "meta": {"image_id": str(index), "design_id": "design", "attack_family": "inpainting" if index % 2 else "real"}}
        if masks:
            row.update(mask=torch.full((1, 2, 2), float(index % 2)), valid=torch.ones(1, 2, 2))
        rows.append(row)
    return rows


@pytest.mark.parametrize("count,expected_steps", [(5, 1), (7, 2), (11, 2)])
def test_partial_accumulation_matches_full_window_gradient(pilots, count, expected_steps):
    torch.manual_seed(4)
    accumulated = TinyPilot()
    expected = copy.deepcopy(accumulated)
    actual_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.07)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.07)
    rows = samples(count)
    loader = DataLoader(rows, batch_size=3)
    result = pilots.train_pilot_epoch(accumulated, loader, actual_optimizer, None, "R", "cpu")
    for window in (rows[:6], rows[6:]):
        if not window:
            continue
        expected_optimizer.zero_grad()
        logits = expected(torch.stack([row["query"] for row in window]))["class_logits"]
        nn.functional.cross_entropy(logits, torch.tensor([row["label"] for row in window])).backward()
        expected_optimizer.step()
    for actual, wanted in zip(accumulated.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-6)
    assert result["n_images"] == count
    assert result["optimizer_steps"] == expected_steps


def test_validation_is_sample_weighted_and_masks_are_not_needed(pilots):
    model = TinyPilot("N")
    rows = samples(5)
    expected_logits = model(torch.stack([row["query"] for row in rows]))["attack_logit"]
    expected_loss = nn.functional.binary_cross_entropy_with_logits(expected_logits, torch.tensor([0., 1., 0., 1., 0.])).item()
    summary, predictions = pilots.evaluate_pilot(model, DataLoader(rows, batch_size=2), "N", "cpu")
    assert summary["image_loss"] == pytest.approx(expected_loss)
    assert summary["n_images"] == 5
    assert summary["subgroups"]["copy"]["auc"] is None
    assert summary["subgroups"]["copy"]["n_altered"] == 0
    assert len(predictions) == 5
    assert "attack_logit" in predictions[0]
    assert "bonafide_score" in predictions[0]


def test_dense_targets_only_used_for_n_training(pilots):
    for track in ("R", "D"):
        model = TinyPilot(track)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        stats = pilots.train_pilot_epoch(model, DataLoader(samples(), batch_size=2), optimizer, None, track, "cpu")
        assert stats["bce_map"] is None
    model = TinyPilot("N")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises((ValueError, KeyError), match="mask"):
        pilots.train_pilot_epoch(model, DataLoader(samples(), batch_size=2), optimizer, None, "N", "cpu")
    stats = pilots.train_pilot_epoch(model, DataLoader(samples(masks=True), batch_size=2), optimizer, None, "N", "cpu")
    assert stats["bce_map"] > 0
    assert stats["loss"] > stats["image_loss"]


def test_d_rejects_batched_evaluation(pilots):
    with pytest.raises(ValueError, match="batch.*one|batch.*1"):
        pilots.evaluate_pilot(TinyPilot("D"), DataLoader(samples(), batch_size=2), "D", "cpu")


def test_optimizer_coverage_rejects_duplicate_and_omitted_parameters(pilots):
    model = TinyPilot()
    assert pilots.optimizer_coverage(model)["passed"]
    model.parameter_groups = lambda: (list(model.encoder.parameters()), list(model.parameters()))
    with pytest.raises(ValueError, match="coverage|duplicate"):
        pilots.optimizer_coverage(model)
    model.parameter_groups = lambda: ([], list(model.head.parameters()))
    with pytest.raises(ValueError, match="coverage|missing"):
        pilots.optimizer_coverage(model)


def test_initialization_provenance_rejects_random_and_hash_mismatch(pilots):
    model = TinyPilot()
    assert pilots.validate_initialized_model(model, "R", "a" * 64)["passed"]
    with pytest.raises(ValueError, match="hash|SHA256"):
        pilots.validate_initialized_model(model, "R", "c" * 64)
    model.initialization_provenance["requested_initialization"] = "scratch"
    with pytest.raises(ValueError, match="initialization"):
        pilots.validate_initialized_model(model, "R", "a" * 64)


def test_official_missing_bn_counter_migration_is_hash_bound(pilots):
    model = TinyPilot()
    digest = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
    model.initialization_provenance.update(checkpoint_sha256=digest, missing_keys=["bn.num_batches_tracked"],
                                          synthesized_keys=["bn.num_batches_tracked"],
                                          synthesized_values={"bn.num_batches_tracked": 0},
                                          migration_policy="official_imagenet1k_v1_missing_bn_counters_zero_v1")
    assert pilots.validate_initialized_model(model, "R", digest)["passed"]
    model.initialization_provenance["checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="missing|migration"):
        pilots.validate_initialized_model(model, "R", "a" * 64)


def test_r_recipe_never_clips_gradients(pilots):
    model = TinyPilot()
    expected = copy.deepcopy(model)
    rows = samples(2)
    for row in rows:
        row["query"] *= 1000
    actual_optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=.001)
    pilots.train_pilot_epoch(model, DataLoader(rows, batch_size=2), actual_optimizer, None, "R", "cpu")
    expected_optimizer.zero_grad()
    nn.functional.cross_entropy(expected(torch.stack([row["query"] for row in rows]))["class_logits"], torch.tensor([0, 1])).backward()
    expected_optimizer.step()
    for actual, wanted in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, wanted)


def test_evaluation_rejects_mutated_buffers(pilots):
    class Mutating(TinyPilot):
        def __init__(self):
            super().__init__()
            self.register_buffer("calls", torch.zeros(()))
        def forward(self, query):
            self.calls.add_(1)
            return super().forward(query)
    with pytest.raises(ValueError, match="buffers"):
        pilots.evaluate_pilot(Mutating(), DataLoader(samples(), batch_size=2), "R", "cpu")


def test_overfit_selection_fixed_composition_sorted(pilots):
    rows = pd.DataFrame([{"image_id": f"{family}-{index:02d}", "attack_family": family, "label": int(family != "real")}
                         for family, count in (("real", 10), ("inpainting", 8), ("copy", 3)) for index in range(count)])
    selected = pilots.select_phase_rows(rows.sample(frac=1, random_state=9), "overfit", "train")
    assert selected["attack_family"].value_counts().to_dict() == {"real": 8, "inpainting": 6, "copy": 2}
    assert selected.image_id.tolist() == sorted(selected.image_id.tolist())
    with pytest.raises(ValueError, match="test|role"):
        pilots.select_phase_rows(rows, "smoke", "test")


def test_builder_preserves_external_random_streams(pilots, monkeypatch, tmp_path):
    weights = tmp_path / "offline.pth"
    weights.write_bytes(b"fixture")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    def builder(*args, **kwargs):
        random.random()
        np.random.rand()
        model = TinyPilot()
        model.initialization_provenance["checkpoint_sha256"] = digest
        return model
    monkeypatch.setattr(pilots, "_model_builder", lambda track: builder)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    pilots.build_initialized_model({"track": "R", "seed": 42}, weights, digest)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_cli_forbids_training_checkpoint_before_any_data_access(pilots, tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        pilots.main(["--config", "missing.yaml", "--raw-root", str(tmp_path), "--run-dir", str(tmp_path / "train"),
                     "--phase", "train", "--weights", "offline.pth", "--weights-sha256", "a" * 64, "--checkpoint", "probe.pt"])


def test_runtime_rejects_undeclared_official_weight_identity(pilots):
    with pytest.raises(ValueError, match="official|declared"):
        pilots.validate_weight_request("R", "a" * 64)
    pilots.validate_weight_request("R", "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a")
    pilots.validate_weight_request("N", "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618")


def test_n_declares_warn_only_determinism_without_weakening_fp32(pilots):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        precision = pilots.configure_precision("cpu", "N")
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert precision["deterministic_warn_only"]
        assert precision["known_nondeterministic_ops"] == ["grid_sample_cuda_backward"]
        assert not precision["training_bit_reproducible"]
        assert not precision["amp"] and not precision["cudnn_tf32"] and not precision["matmul_tf32"]
        pilots.configure_precision("cpu", "R")
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        probe = pilots.probe_cuda_grid_sample_policy("cpu", "N")
        assert not probe["performed"]
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_selection_keeps_earlier_epoch_on_exact_ties(pilots):
    assert pilots.selection_key("accuracy", .75, .8, 1., 2.) == (.75, .8)
    assert not pilots.selection_key("accuracy", .75, .8, .1, .1) > (.75, .8)


class FixtureDataset(torch.utils.data.Dataset):
    def __init__(self, rows, raw_root, legacy_root, track, train, augment):
        self.rows = rows.reset_index(drop=True)
        self.track = track
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        item = {"query": torch.tensor([float(row.label * 2 - 1), 1.]), "label": int(row.label),
                "meta": {key: str(row[key]) for key in ("image_id", "design_id", "attack_family")}}
        if self.track == "N" and self.train:
            item.update(mask=torch.full((1, 2, 2), float(row.label)), valid=torch.ones(1, 2, 2))
        return item


def fixture_rows(role):
    return pd.DataFrame([dict(image_id=f"{role}-{index:02d}", path=f"{role}-{index}.png", label=index % 2,
                              design_id="design", component_id=f"component-{index}", role=role,
                              attack_family="real" if index % 2 == 0 else ("copy" if index % 8 == 1 else "inpainting"))
                         for index in range(32)])


@pytest.fixture
def runtime_fixture(pilots, monkeypatch, tmp_path):
    from refer_or_clear.data import query
    config = tmp_path / "config.yaml"
    config.write_text("track: R\n")
    records, split, weights = (tmp_path / name for name in ("records.csv", "split.csv", "weights.pth"))
    records.write_text("fixture records")
    split.write_text("fixture split")
    weights.write_bytes(b"fixture weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setattr(pilots, "OFFICIAL_WEIGHT_SHA256", {"R": digest}, raising=False)
    def builder(**kwargs):
        model = TinyPilot()
        model.initialization_provenance["checkpoint_sha256"] = digest
        model.initialization_provenance["checkpoint_path"] = str(weights)
        return model
    opened_roles = []
    def load_rows(records_csv, split_csv, role):
        assert role != "test", "test role must never be opened"
        opened_roles.append(role)
        return fixture_rows(role)
    monkeypatch.setattr(query, "AnchorQueryDataset", FixtureDataset)
    monkeypatch.setattr(query, "load_role_rows", load_rows)
    monkeypatch.setattr(pilots, "_model_builder", lambda track: builder)
    return dict(config_path=config, raw_root=tmp_path, legacy_root=None, device="cpu", weights=weights,
                weights_sha256=digest, records=records, split=split), opened_roles


def test_smoke_runs_real_recipe_once_and_never_opens_test(pilots, runtime_fixture, tmp_path):
    options, opened_roles = runtime_fixture
    result = pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="smoke")
    assert result["epochs_completed"] == 1
    assert opened_roles == ["train", "val"]
    assert result["n_images"] == {"train": 32, "val": 32}
    assert result["optimizer"]["name"] == "sgd"
    assert result["inference_before"]["passed"]
    assert result["inference_after"]["passed"]
    assert (tmp_path / "smoke" / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "smoke" / "val_predictions.csv").is_file()
    with pytest.raises(ValueError, match="nonempty|empty"):
        pilots.run_pilot(**options, run_dir=tmp_path / "smoke", phase="smoke")


def test_single_class_smoke_keeps_missing_auc_null(pilots, runtime_fixture, monkeypatch, tmp_path):
    options, _ = runtime_fixture
    from refer_or_clear.data import query
    def rows(records, split, role):
        frame = fixture_rows(role)
        if role == "val":
            frame["label"] = 0
            frame["attack_family"] = "real"
        return frame
    monkeypatch.setattr(query, "load_role_rows", rows)
    result = pilots.run_pilot(**options, run_dir=tmp_path / "single-class", phase="smoke")
    summary = pilots._read_json(tmp_path / "single-class" / "val_summary.json")
    assert summary["auc"] is None
    assert result["selection_key"] == [summary["accuracy"], None]


def test_overfit_uses_common_classification_criterion_not_dense_loss(pilots, runtime_fixture, tmp_path):
    options, opened_roles = runtime_fixture
    result = pilots.run_pilot(**options, run_dir=tmp_path / "overfit", phase="overfit")
    assert opened_roles == ["train"]
    assert result["updates"] <= 500
    assert result["updates"] % 50 == 0
    assert result["overfit"]["n_correct"] >= 15
    assert result["overfit"]["image_loss"] < .15
    assert result["passed"]
    assert result["optimizer"]["name"] == "adamw"


def test_train_rejects_non100_epochs_before_model_construction(pilots, runtime_fixture, tmp_path):
    options, _ = runtime_fixture
    options["config_path"].write_text("track: R\nepochs: 2\n")
    with pytest.raises(ValueError, match="100"):
        pilots.run_pilot(**options, run_dir=tmp_path / "train", phase="train")


def test_train_only_accepts_completed_pretrain_gate(pilots):
    identities = {"source_sha256": "a" * 64}
    initialization = {"state_sha256": "b" * 64, "buffer_sha256": "c" * 64, "pretrained_tensor_sha256": "d" * 64}
    gate = {"schema_version": "anchor_pilot_v1", "phase": "pretrain", "passed": True, "failures": [],
            "identities": identities, "fresh_initialization": initialization,
            "evidence": {"overfit": {"passed": True}, "smoke": {"passed": True}}}
    pilots.validate_pretrain_initialization(gate, identities, initialization)
    for key, value in (("phase", "overfit"), ("schema_version", "legacy"), ("passed", False), ("evidence", {})):
        with pytest.raises(ValueError, match="pretrain|diagnostic"):
            pilots.validate_pretrain_initialization({**gate, key: value}, identities, initialization)
    with pytest.raises(ValueError, match="initialization"):
        pilots.validate_pretrain_initialization(gate, identities, {**initialization, "state_sha256": "e" * 64})


def test_overfit_and_replay_use_real_training_role_dataset(pilots, runtime_fixture, monkeypatch, tmp_path):
    options, _ = runtime_fixture
    from refer_or_clear.data import query
    monkeypatch.setattr(query, "AnchorQueryDataset", RealAnchorQueryDataset)
    for role in ("train", "val"):
        for row in fixture_rows(role).to_dict("records"):
            cv2.imwrite(str(tmp_path / row["path"]), np.full((7, 11, 3), 255 * row["label"], dtype=np.uint8))
    class ImagePilot(TinyPilot):
        def forward(self, query):
            return super().forward(query[:, :2].mean((2, 3)))
    def builder(**kwargs):
        model = ImagePilot()
        model.initialization_provenance.update(checkpoint_sha256=options["weights_sha256"], checkpoint_path=str(options["weights"]))
        return model
    monkeypatch.setattr(pilots, "_model_builder", lambda track: builder)
    run_dir = tmp_path / "real-overfit"
    result = pilots.run_pilot(**options, run_dir=run_dir, phase="overfit")
    assert result["passed"]
    replay = pilots.run_pilot(**options, run_dir=run_dir, phase="replay")
    assert replay["passed"]
