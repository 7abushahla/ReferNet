import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear import pilots


@pytest.fixture
def diagnostic():
    path = ROOT / "scripts/check_anchor_d_context.py"
    assert path.is_file(), "isolated D context diagnostic is not implemented"
    spec = importlib.util.spec_from_file_location("anchor_d_context_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticD(nn.Module):
    track = "D"
    architecture_id = "anchor_direct_ia18_v1"

    def __init__(self):
        super().__init__()
        self.backbone = nn.BatchNorm2d(4, track_running_stats=False)
        self.classifier = nn.Linear(4, 1)
        self.register_buffer("sentinel", torch.tensor(7.))
        with torch.no_grad():
            self.classifier.weight.fill_(1.)
            self.classifier.bias.fill_(.2)
        self.calls = []

    def _check_bn_policy(self):
        assert not self.training and not self.backbone.training
        assert not self.backbone.track_running_stats
        assert self.backbone.running_mean is None

    def forward(self, query):
        if not self.training and len(query) != 1:
            raise ValueError("D batch-statistics evaluation requires batch size one")
        self.calls.append(len(query))
        return {"attack_logit": self.classifier(self.backbone(query).mean((2, 3))).squeeze(1)}


@pytest.fixture
def probe():
    queries = torch.arange(16 * 4 * 3 * 3, dtype=torch.float32).reshape(16, 4, 3, 3) / 100
    labels = torch.tensor([0] * 8 + [1] * 8)
    identifiers = [f"probe_{index:02}" for index in range(16)]
    return queries, labels, identifiers


def synthetic_encoder(backbone, queries):
    assert not backbone.training
    assert torch.is_inference_mode_enabled()
    return [backbone(queries)]


def test_context_pairs_exact_groups_and_preserves_public_guard(diagnostic, probe):
    model = SyntheticD().eval()
    before = pilots.tensor_sha256(model.state_dict())
    result = diagnostic.evaluate_context(model, *probe, pilots=pilots, encode_query=synthetic_encoder)
    assert result["batch_sizes"] == [1, 4, 8, 16]
    assert result["singleton_equivalence"]["passed"]
    assert result["fingerprints_before"] == result["fingerprints_after"]
    assert pilots.tensor_sha256(model.state_dict()) == before
    assert model.calls == [1] * 16
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())
    assert len(result["predictions"]) == 64
    for size in (1, 4, 8, 16):
        records = [row for row in result["predictions"] if row["batch_size"] == size]
        assert [row["image_id"] for row in records] == probe[2]
        for index, record in enumerate(records):
            start = index // size * size
            assert record["group_members"] == probe[2][start:start + size]
            assert record["position"] == index % size
            assert record["singleton_logit"] == result["predictions"][index]["attack_logit"]
        logits = torch.tensor([row["attack_logit"] for row in records])
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, probe[1].float(), reduction="none")
        assert [row["image_ce"] for row in records] == pytest.approx(losses.tolist())
        assert result["summaries"][str(size)]["image_ce"] == pytest.approx(losses.mean().item())
    assert any(abs(row["delta_from_singleton"]) > .1 for row in result["predictions"] if row["batch_size"] == 8)
    with pytest.raises(ValueError, match="batch size one"):
        model(probe[0][:4])


def test_singleton_mismatch_stops_before_larger_batches(diagnostic, probe):
    batches = []

    def wrong_encoder(backbone, queries):
        batches.append(len(queries))
        return [backbone(queries) + 1]

    with pytest.raises(ValueError, match="singleton equivalence"):
        diagnostic.evaluate_context(SyntheticD().eval(), *probe, pilots=pilots, encode_query=wrong_encoder)
    assert batches and set(batches) == {1}


@pytest.mark.parametrize("target", ["weight", "buffer"])
def test_mutation_fails_closed(diagnostic, probe, target):
    model = SyntheticD().eval()

    def mutating_encoder(backbone, queries):
        if target == "weight":
            model.classifier.weight.add_(1)
        else:
            model.sentinel.add_(1)
        return [backbone(queries)]

    with pytest.raises(ValueError, match="immutable"):
        diagnostic.evaluate_context(model, *probe, pilots=pilots, encode_query=mutating_encoder)


@pytest.mark.parametrize("change", ["duplicate", "count", "dtype", "labels", "track", "running_bn"])
def test_probe_contract_rejects_invalid_inputs(diagnostic, probe, change):
    queries, labels, identifiers = probe
    model = SyntheticD().eval()
    if change == "duplicate":
        identifiers = [identifiers[0]] * 16
    elif change == "count":
        queries, labels, identifiers = queries[:8], labels[:8], identifiers[:8]
    elif change == "dtype":
        queries = queries.double()
    elif change == "labels":
        labels = labels + 2
    elif change == "track":
        model.track = "N"
    else:
        model.backbone = nn.BatchNorm2d(4).eval()
    with pytest.raises(ValueError):
        diagnostic.evaluate_context(model, queries, labels, identifiers, pilots=pilots, encode_query=synthetic_encoder)


@pytest.fixture
def bound_inputs(tmp_path, diagnostic, monkeypatch):
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    source_path = source / "src/science.py"
    source_path.write_text("frozen science\n")
    files = [{"path": "src/science.py", "size": source_path.stat().st_size, "sha256": pilots.file_sha256(source_path)}]
    runtime_hash = pilots.json_sha256({entry["path"]: entry["sha256"] for entry in files})
    monkeypatch.setattr(diagnostic, "FROZEN_SOURCE_SHA256", runtime_hash)
    config = {"schema_version": "anchor_pilot_v1", "track": "D", "architecture_id": "anchor_direct_ia18_v1",
              "seed": 42, "fold": 0, "eval_batch_size": 1, "bn_policy": "batch_stats"}
    paths = {}
    for name in ("config", "records", "split", "weights", "checkpoint", "predictions"):
        paths[name] = tmp_path / name
        paths[name].write_text(name)
    rows = pd.DataFrame([{"image_id": f"probe_{index:02}", "label": int(index >= 8), "design_id": "sidtd/example",
                          "attack_family": "" if index < 8 else "text_inpaint" if index < 14 else "text_copy",
                          "component_id": f"component_{index}", "path": f"raw/{index}.jpg", "role": "train"}
                         for index in range(16)])
    legacy = tmp_path / "legacy"
    raw = tmp_path / "raw"
    raw.mkdir()
    legacy.mkdir()
    data_files = []
    for relative in [f"canvases/{identifier}.png" for identifier in rows.image_id] + ["zones/sidtd_example.png"]:
        path = legacy / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(relative.encode())
        data_files.append({"kind": "legacy", "path": relative, "sha256": pilots.file_sha256(path), "size": path.stat().st_size})
    identities = {"source_sha256": runtime_hash, "config_sha256": pilots.json_sha256(config),
                  **{f"{key}_sha256": pilots.file_sha256(paths[key]) for key in ("records", "split", "weights")}}
    train = {"schema_version": "anchor_pilot_v1", "architecture_id": config["architecture_id"], "track": "D",
             "phase": "overfit", "config": config, "identities": identities, "opened_roles": ["train"],
             "no_test_open": True, "n_images": {"train": 16}, "membership": {"train": pilots._rows_manifest(rows)},
             "phase_recipe": {"batch_size": 8, "drop_last": False, "augment": {"enabled": False}, "accumulation_steps": 1},
             "updates": 500, "selected_checkpoint_sha256": pilots.file_sha256(paths["checkpoint"])}
    manifest = {"schema_version": "anchor_job_manifest_v1", "track": "D", "phase": "diagnostic",
                "allocation": {"job_id": "37963"}, "data_scope": "train_val_only", "effective_config": config,
                "runtime_identities": identities, "source_files": files, "data_files": data_files,
                "membership": {"train": rows.image_id.tolist(), "val": ["unopened_val"]},
                "provenance": {**identities, "source_sha256": pilots.json_sha256(files),
                               "config_sha256": pilots.file_sha256(paths["config"]), "data_sha256": pilots.json_sha256(data_files)}}
    authorities = tmp_path / "authorities"
    authorities.mkdir()
    for name, payload in (("parent_manifest", manifest), ("train_json", train)):
        paths[name] = authorities / f"{name}.json"
        pilots.write_json(paths[name], payload)
    args = SimpleNamespace(**paths, source_dir=source, raw_root=raw, legacy_root=legacy, run_dir=tmp_path / "report",
                           device="cpu", workers=0, parent_manifest_sha256=pilots.file_sha256(paths["parent_manifest"]),
                           train_json_sha256=pilots.file_sha256(paths["train_json"]),
                           predictions_sha256=pilots.file_sha256(paths["predictions"]))
    monkeypatch.setattr(diagnostic, "FROZEN_CHECKPOINT_SHA256", train["selected_checkpoint_sha256"])
    runtime = SimpleNamespace(**{name: getattr(pilots, name) for name in
                                 ("file_sha256", "json_sha256", "_rows_manifest", "select_phase_rows")},
                              source_sha256=lambda: runtime_hash,
                              validate_weight_request=lambda track, digest: None)
    return args, runtime, config, rows, manifest, train


def test_bound_inventory_opens_only_probe_canvases_and_zone(diagnostic, bound_inputs, monkeypatch):
    args, runtime, config, rows, _, _ = bound_inputs
    original = runtime.file_sha256
    opened = []

    def record_hash(path):
        opened.append(Path(path))
        return original(path)

    runtime.file_sha256 = record_hash
    result = diagnostic.validate_bindings(args, runtime, config, rows)
    assert result["membership"] == pilots._rows_manifest(rows)
    assert len(result["verified_data_files"]) == 17
    assert not any(path.is_relative_to(args.raw_root) for path in opened)
    assert not any("unopened_val" in str(path) for path in opened)


@pytest.mark.parametrize("change", ["parent_hash", "train_hash", "checkpoint", "source", "source_file", "records", "split", "weights",
                                    "config", "membership", "order", "train_role", "parent_val", "parent_train", "recipe",
                                    "data_digest", "canvas", "zone", "missing_canvas", "duplicate_data", "unsafe_path"])
def test_binding_tampering_fails_closed(diagnostic, bound_inputs, change):
    args, runtime, config, rows, manifest, train = bound_inputs
    if change == "parent_hash":
        args.parent_manifest_sha256 = "0" * 64
    elif change == "train_hash":
        args.train_json_sha256 = "0" * 64
    elif change in ("checkpoint", "records", "split", "weights"):
        getattr(args, change).write_text("tampered")
    elif change == "source":
        runtime.source_sha256 = lambda: "0" * 64
    elif change == "source_file":
        (args.source_dir / "src/science.py").write_text("tampered")
    elif change == "config":
        config = {**config, "seed": 41}
    elif change == "membership":
        train["membership"]["train"][0]["label"] = 1
    elif change == "order":
        train["membership"]["train"].reverse()
    elif change == "train_role":
        rows = rows.copy()
        rows.loc[0, "role"] = "val"
    elif change == "parent_val":
        manifest["membership"]["val"].append(rows.image_id.iloc[0])
    elif change == "parent_train":
        manifest["membership"]["train"].remove(rows.image_id.iloc[0])
    elif change == "recipe":
        train["phase_recipe"]["augment"]["enabled"] = True
    elif change == "data_digest":
        manifest["provenance"]["data_sha256"] = "0" * 64
    elif change in ("canvas", "zone"):
        entry = manifest["data_files"][0 if change == "canvas" else -1]
        (args.legacy_root / entry["path"]).write_bytes(b"tampered")
    elif change == "missing_canvas":
        manifest["data_files"].pop(0)
        manifest["provenance"]["data_sha256"] = pilots.json_sha256(manifest["data_files"])
    elif change == "duplicate_data":
        manifest["data_files"].append(copy.deepcopy(manifest["data_files"][0]))
        manifest["provenance"]["data_sha256"] = pilots.json_sha256(manifest["data_files"])
    else:
        rows = rows.copy()
        rows.loc[0, "image_id"] = "../escape"
        train["membership"]["train"] = pilots._rows_manifest(rows)
        manifest["membership"]["train"] = rows.image_id.tolist()
    for name, payload in (("parent_manifest", manifest), ("train_json", train)):
        if change not in ("parent_hash", "train_hash"):
            pilots.write_json(getattr(args, name), payload)
            setattr(args, f"{name}_sha256", pilots.file_sha256(getattr(args, name)))
    with pytest.raises((ValueError, FileNotFoundError)):
        diagnostic.validate_bindings(args, runtime, config, rows)


def test_cli_requires_explicit_authorities_and_fixed_batch_protocol(diagnostic):
    with pytest.raises(SystemExit):
        diagnostic.parse_args([])
    help_text = diagnostic.argument_parser().format_help()
    assert "--source-dir" in help_text and "--train-json-sha256" in help_text
    assert "--parent-manifest-sha256" in help_text
    assert "--batch-size" not in help_text
    assert "--epochs" not in help_text


@pytest.mark.parametrize("change", ["order", "label", "logit", "missing"])
def test_saved_singleton_replay_fails_before_context_batches(diagnostic, probe, change):
    model = SyntheticD().eval()
    with torch.inference_mode():
        expected = [{"image_id": identifier, "label": int(probe[1][index]),
                     "attack_logit": float(model(probe[0][index:index + 1])["attack_logit"][0])}
                    for index, identifier in enumerate(probe[2])]
    if change == "order":
        expected.reverse()
    elif change == "label":
        expected[0]["label"] = 1
    elif change == "logit":
        expected[0]["attack_logit"] += 1
    else:
        expected.pop()
    batches = []

    def observed_encoder(backbone, queries):
        batches.append(len(queries))
        return synthetic_encoder(backbone, queries)

    with pytest.raises(ValueError, match="singleton"):
        diagnostic.evaluate_context(model, *probe, pilots=pilots, encode_query=observed_encoder, expected_predictions=expected)
    assert batches == [1] * 16


def test_fresh_runtime_rejects_live_import_contamination(diagnostic, tmp_path):
    (tmp_path / "src").mkdir()
    with pytest.raises(ValueError, match="fresh process"):
        diagnostic.load_runtime(tmp_path)


@pytest.fixture
def synthetic_run(diagnostic, bound_inputs, probe, monkeypatch):
    from refer_or_clear.data import query
    from refer_or_clear.models import anchor_changer
    from refer_or_clear import pilot_config

    args, runtime, config, rows, manifest, train = bound_inputs
    queries, labels, identifiers = probe
    assert identifiers == rows.image_id.tolist()
    model = SyntheticD().eval()
    model.initialization_provenance = {"checkpoint_path": "/original/weights.pth", "checkpoint_sha256": runtime.file_sha256(args.weights)}
    train["initialization"] = {"provenance": model.initialization_provenance}
    pilots.save_pilot_checkpoint(args.checkpoint, model, config, train["identities"], 500, (), "overfit")
    train["selected_checkpoint_sha256"] = pilots.file_sha256(args.checkpoint)
    monkeypatch.setattr(diagnostic, "FROZEN_CHECKPOINT_SHA256", train["selected_checkpoint_sha256"])
    with torch.inference_mode():
        predictions = [{"image_id": identifier, "label": int(labels[index]),
                        "attack_logit": float(model(queries[index:index + 1])["attack_logit"][0])}
                       for index, identifier in enumerate(identifiers)]
        logits = torch.tensor([row["attack_logit"] for row in predictions])
        train["overfit"] = {"n_correct": int(((logits.sigmoid() > .5).long() == labels).sum()),
                            "image_loss": pilots._image_loss({"attack_logit": logits}, labels, "D").item()}
    pilots._write_csv(args.predictions, predictions)
    pilots.write_json(args.train_json, train)
    args.train_json_sha256 = pilots.file_sha256(args.train_json)
    args.predictions_sha256 = pilots.file_sha256(args.predictions)
    events = []

    class SyntheticDataset:
        def __init__(self, actual_rows, raw_root, legacy_root, track, training_role, augmentation):
            events.append("dataset")
            assert actual_rows.equals(rows)
            assert (track, training_role, augmentation) == ("D", True, {"enabled": False})

        def __len__(self):
            return 16

        def __getitem__(self, index):
            row = rows.iloc[index]
            return {"query": queries[index], "label": int(labels[index]),
                    "meta": {key: row[key] for key in ("image_id", "design_id", "attack_family")}}

    def build(config, weights, digest):
        events.append("initialization")
        initialized = SyntheticD()
        initialized.initialization_provenance = {**model.initialization_provenance, "checkpoint_path": str(weights)}
        return initialized

    for name in ("read_pilot_checkpoint", "configure_precision", "_loader", "_read_predictions", "ATOL", "RTOL",
                 "tensor_sha256", "_image_loss", "compare_replay", "_write_csv", "write_json"):
        setattr(runtime, name, getattr(pilots, name))
    runtime.build_initialized_model = build
    monkeypatch.setattr(pilot_config, "resolve_pilot_config", lambda path: config)
    monkeypatch.setattr(query, "load_role_rows", lambda records, split, role: rows if role == "train" else pytest.fail("nontrain role"))
    monkeypatch.setattr(query, "AnchorQueryDataset", SyntheticDataset)
    monkeypatch.setattr(anchor_changer, "encode_query", synthetic_encoder)
    return args, runtime, events


def test_synthetic_end_to_end_uses_frozen_helper_checkpoint_and_loader(diagnostic, synthetic_run):
    args, runtime, events = synthetic_run
    protected = {path: pilots.file_sha256(path) for path in
                 (args.parent_manifest, args.train_json, args.checkpoint, args.predictions, args.weights)}
    result = diagnostic.run(args, runtime)
    assert events == ["initialization", "dataset"]
    assert result["saved_singleton_replay"]["passed"]
    assert result["checkpoint_epoch"] == 500
    assert result["scientific_gate_passed"] is None
    assert result["bindings"]["opened_roles"] == ["train"]
    assert result["bindings"]["raw_pixels_read"] is False
    assert result["initialization_checkpoint_path_original"] == "/original/weights.pth"
    assert result["initialization_checkpoint_path_relocated"] == str(args.weights)
    assert (args.run_dir / "context.json").is_file()
    assert len(pilots._read_predictions(args.run_dir / "paired_predictions.csv")) == 64
    assert protected == {path: pilots.file_sha256(path) for path in protected}


@pytest.mark.parametrize("change", ["checkpoint", "metadata", "output", "output_source"])
def test_invalid_run_stops_before_initialization_or_pixels(diagnostic, synthetic_run, change):
    args, runtime, events = synthetic_run
    if change == "checkpoint":
        args.checkpoint.write_bytes(b"tampered")
    elif change == "metadata":
        args.train_json_sha256 = "0" * 64
    elif change == "output":
        args.run_dir.mkdir()
    else:
        args.run_dir = args.source_dir / "output"
    with pytest.raises(ValueError):
        diagnostic.run(args, runtime)
    assert events == []
