import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import statistics
from types import SimpleNamespace
from contextlib import contextmanager

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bench_anchor_candidates.py"
WORKER = ROOT / "scripts/bench_anchor_candidate_worker.py"
TRACKS = ("R", "D_running_pilot", "N")
REVISIONS = {
    "R": "0959335bd9a6a22510f6706c49c62d84560cd524",
    "N": "0959335bd9a6a22510f6706c49c62d84560cd524",
    "D_running_pilot": "bef7544a64f36a166f76eeef0d3130e353bd3203",
}
ARCHIVES = {
    "R": "4413528c1afd15f6c420763287bc57d9eb523df2a69138d5b472f0b4b2092ba4",
    "N": "4413528c1afd15f6c420763287bc57d9eb523df2a69138d5b472f0b4b2092ba4",
    "D_running_pilot": "00d0b99e635536d27a01c6f4f39bd94e14e98409dc5616c722f7dd4db3f8d5de",
}
ARCHITECTURES = {
    "R": "anchor_rgb_r50_v1", "D_running_pilot": "anchor_direct_running_ia18_pilot_v1",
    "N": "anchor_absent_ia18_v1",
}
WEIGHTS = {
    "R": "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    "D_running_pilot": "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618",
    "N": "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def authority(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value))
    return authority(path)


@pytest.fixture
def bench():
    assert SCRIPT.is_file(), "candidate metadata preflight is not implemented"
    spec = importlib.util.spec_from_file_location("candidate_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def catalogue(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    rows = []
    split_rows = []
    for index in range(2222):
        row = {"image_id": f"id{index:04d}", "label": index % 2, "design_id": "design",
               "component_id": f"component{index}", "path": f"raw/id{index:04d}.jpg",
               "attack_family": "genuine" if index % 2 == 0 else "fraud"}
        rows.append(row)
        role = "train" if index < 1778 else "val" if index < 2000 else "test"
        split_rows.append({**{key: row[key] for key in ("image_id", "label", "design_id", "component_id")},
                           "role": role, "protocol": "A", "ref_group": "it0"})
    for filename, content in (("records.csv", rows), ("split.csv", split_rows)):
        with (metadata / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    tools = tmp_path / "measurement"
    tools.mkdir()
    weight_directory = tmp_path / "weights"
    weight_directory.mkdir()
    tool_hashes = {}
    for filename in ("bench_anchor_pilots.py", "report_anchor_pilots.py"):
        path = tools / filename
        path.write_text("raise RuntimeError('must not import measurement helpers in prepare')\n")
        tool_hashes[filename] = authority(path)["sha256"]
    candidates = []
    for track in TRACKS:
        source = tmp_path / f"source-{track}"
        source.mkdir()
        run = tmp_path / f"run-{track}"
        run.mkdir()
        pilot = run / "pilot"
        pilot.mkdir()
        raw = tmp_path / f"raw-{track}"
        legacy = tmp_path / f"legacy-{track}"
        weights = weight_directory / f"weights-{track}.pth"
        config = {"schema_version": "anchor_pilot_v1", "track": track,
                  "architecture_id": ARCHITECTURES[track], "fold": 0,
                  "seed": 42, "epochs": 100, "reference": "none",
                  "data_contract": "raw_rgb_299_v1" if track == "R" else "legacy_geometry_diagnostic"}
        source_files = [{"path": "src/refer_or_clear/pilots.py", "sha256": digest(REVISIONS[track]), "size": 1},
                        {"path": "scripts/check_anchor_pilots.py", "sha256": digest("check"), "size": 1}]
        identities = {"source_sha256": digest({item["path"]: item["sha256"] for item in source_files}),
                      "config_sha256": digest(config), "weights_sha256": WEIGHTS[track],
                      "records_sha256": authority(metadata / "records.csv")["sha256"],
                      "split_sha256": authority(metadata / "split.csv")["sha256"]}
        paths = {"source_dir": str(source), "run_dir": str(run), "pilot_dir": str(pilot),
                 "weights": str(weights), "raw_root": str(raw), "legacy_root": str(legacy),
                 "records": str(metadata / "records.csv"), "split": str(metadata / "split.csv")}
        membership = {role: [item["image_id"] for item in split_rows if item["role"] == role]
                      for role in ("train", "val")}
        parent = {"schema_version": "anchor_job_manifest_v1", "phase": "train", "track": track,
                  "effective_config": config, "runtime_identities": identities, "paths": paths,
                  "membership": membership, "source_files": source_files,
                  "provenance": {"source_sha256": digest(source_files),
                                 **{key: value for key, value in identities.items() if key != "source_sha256"}}}
        candidates.append({"track": track, "architecture_id": ARCHITECTURES[track], "status": "pending",
                           "run_dir": str(pilot), "parent_manifest": write_json(run / "input_manifest.json", parent),
                           "scientific_source": {"directory": str(source), "revision": REVISIONS[track],
                                                 "archive_sha256": ARCHIVES[track],
                                                 "inventory_sha256": digest(source_files)},
                           "identities": identities, "weights": {"path": str(weights), "sha256": WEIGHTS[track]},
                           "records": authority(metadata / "records.csv"),
                           "split": authority(metadata / "split.csv"), "raw_root": str(raw),
                           "legacy_root": str(legacy), "train_json": None, "selected_checkpoint": None,
                           "selected_evidence": None, "completion_review": None})
    return {"schema_version": "anchor_candidate_bindings_v1", "protocol": "A", "fold": 0,
            "seed": 42, "role": "train", "image_ids": membership["train"][:200],
            "measurement_tools": {"directory": str(tools), "files": tool_hashes}, "candidates": candidates}


def complete(candidate):
    parent = json.loads(Path(candidate["parent_manifest"]["path"]).read_text())
    with Path(candidate["records"]["path"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    membership = {}
    for role in ("train", "val"):
        role_ids = set(parent["membership"][role])
        membership[role] = [{**row, "label": int(row["label"])} for row in rows if row["image_id"] in role_ids]
    pilot = Path(candidate["run_dir"])
    candidate["status"] = "complete"
    candidate["selected_checkpoint"] = {"path": str(pilot / "checkpoints/best.pt"),
                                        "sha256": digest(candidate["track"] + "checkpoint")}
    train = {"schema_version": "anchor_pilot_v1", "phase": "train", "track": candidate["track"],
             "architecture_id": candidate["architecture_id"], "config": parent["effective_config"],
             "identities": candidate["identities"], "paths": parent["paths"], "membership": membership,
             "epochs_completed": 100, "selected_epoch": 55, "selection_key": [0.9, 0.95],
             "selected_checkpoint_sha256": candidate["selected_checkpoint"]["sha256"],
             "passed": True, "opened_roles": ["train", "val"], "no_test_open": True,
             "n_images": {"train": 1778, "val": 222}}
    candidate["train_json"] = write_json(pilot / "train.json", train)
    candidate["selected_evidence"] = {}
    for filename in ("val_summary.json", "val_predictions.csv", "log.csv", "gate.json", "replay.json"):
        path = pilot / filename
        path.write_text("synthetic selected evidence, never interpreted as a gate\n")
        candidate["selected_evidence"][filename] = authority(path)
    review = pilot.parent / "final-review.md"
    review.write_text("Synthetic independent final selected-validation review\n")
    candidate["completion_review"] = {
        **authority(review), "scope": "final-selected-validation", "approved": True,
        "reviewer": "synthetic-independent-reviewer",
        "bindings": {"parent_manifest_sha256": candidate["parent_manifest"]["sha256"],
                     "train_json_sha256": candidate["train_json"]["sha256"],
                     "selected_checkpoint_sha256": candidate["selected_checkpoint"]["sha256"],
                     "selected_evidence_sha256": {name: item["sha256"]
                                                  for name, item in candidate["selected_evidence"].items()}}}


def assert_invalid(bench, catalogue, message):
    result = bench.prepare_bindings(catalogue)
    assert result["status"] == "not-comparable", result
    assert result["comparable"] is False
    assert result["execution_authorized"] is False
    assert any(message in error for error in result["errors"]), result


def test_distinct_scientific_sources_wait_without_touching_models(bench, catalogue):
    original = copy.deepcopy(catalogue)
    result = bench.prepare_bindings(catalogue)
    assert result["status"] == "waiting-for-completion"
    assert result["comparable"] is False
    assert result["execution_authorized"] is False
    assert result["waiting"] == list(TRACKS)
    assert result["image_ids"] == catalogue["image_ids"]
    assert result["image_ids_sha256"] == digest(catalogue["image_ids"])
    assert "durations" not in result and "timing" not in result
    assert catalogue == original
    assert len({item["identities"]["source_sha256"] for item in catalogue["candidates"]}) > 1
    assert (catalogue["candidates"][0]["identities"]["source_sha256"]
            == catalogue["candidates"][2]["identities"]["source_sha256"])


def test_authentic_r37967_config_without_protocol_preserves_original_identity(bench):
    config = {
        "accumulation_steps": 2, "architecture_id": "anchor_rgb_r50_v1",
        "augment": {"brightness_contrast_p": 0.2, "brightness_offset": [-0.2, 0.2],
                    "brightness_units": "255", "clip": "uint8_0_255", "contrast_factor": [0.8, 1.2],
                    "contrast_semantics": "multiply_pixels", "crop_attempts": 10,
                    "crop_fallback": "bounded_center", "crop_ratio": [0.75, 1.3333333333333333],
                    "crop_ratio_sampling": "log_uniform", "crop_scale": [0.8, 1.0], "enabled": True,
                    "horizontal_flip_p": 0.5, "interpolation": "opencv_inter_linear", "jpeg_quality": None,
                    "preset": "raw_rgb_299_v1", "vertical_flip_p": 0.5},
        "batch_size": 32, "bn_policy": "running", "comparison": "candidate_package",
        "data_contract": "raw_rgb_299_v1", "drop_last": False, "epochs": 100,
        "eval_batch_size": 32, "eval_drop_last": False, "fold": 0, "mask_weight": 0.0,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "optimizer": {"lr": 0.01, "momentum": 0.9, "name": "sgd", "nesterov": False, "weight_decay": 0.0},
        "precision": {"amp": False, "cudnn_tf32": False, "dtype": "float32", "matmul_tf32": False},
        "records_csv": "data/processed/records.csv",
        "scheduler": {"cooldown": 0, "eps": 1e-06, "factor": 0.9, "min_lr": 0.0, "mode": "min",
                      "monitor": "val_ce", "name": "reduce_on_plateau", "patience": 1,
                      "threshold": 0.0001, "threshold_mode": "rel"},
        "schema_version": "anchor_pilot_v1", "seed": 42,
        "selection": {"exact_tie": "earlier_epoch", "primary": "accuracy", "tie_break": "auc"},
        "split_csv": "data/processed/splits/protocol_a_it0.csv", "track": "R", "wrong_design_fraction": 0.0,
    }
    original = copy.deepcopy(config)
    original_identity = "2860b05436a0fcf890b61a5124d717be925be95a710be28beb74414fa9d8030b"
    assert "protocol" not in config and digest(config) == original_identity
    bench.validate_config(config, "R")
    assert config == original and digest(config) == original_identity


@pytest.mark.parametrize("completed", [False, True])
def test_config_without_protocol_preflight_preserves_authenticated_metadata(bench, catalogue, completed):
    if completed:
        for candidate in catalogue["candidates"]:
            complete(candidate)
    original = copy.deepcopy(catalogue)
    files = {}
    for candidate in catalogue["candidates"]:
        for name in ("parent_manifest", "train_json"):
            binding = candidate[name]
            if binding is not None:
                path = Path(binding["path"])
                files[path] = path.read_bytes()
                payload = json.loads(files[path])
                config = payload["effective_config"] if name == "parent_manifest" else payload["config"]
                assert "protocol" not in config
                assert digest(config) == candidate["identities"]["config_sha256"]
    result = bench.prepare_bindings(catalogue)
    assert result["status"] == ("metadata-ready" if completed else "waiting-for-completion"), result
    assert result["errors"] == []
    assert catalogue == original
    assert all(path.read_bytes() == content for path, content in files.items())


def test_complete_is_metadata_ready_never_measured_and_does_not_read_checkpoint(bench, catalogue, monkeypatch):
    for candidate in catalogue["candidates"]:
        complete(candidate)
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix not in (".pt", ".pth", ".jpg", ".png"), str(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    result = bench.prepare_bindings(catalogue)
    assert result["status"] == "metadata-ready", result
    assert result["comparable"] is False and result["execution_authorized"] is False
    assert result["verification"]["selected_checkpoint_bytes"] is False
    assert result["verification"]["scientific_source_bytes"] is False
    assert result["verification"]["offline_weight_bytes"] is False
    assert result["full_source_to_decision_latency"] == "not_measured"


@pytest.mark.parametrize("field,value,message", [
    ("track", "D", "track"), ("track", "D_running", "track"),
    ("architecture_id", "anchor_direct_ia18_v1", "architecture"),
    ("status", "failed", "failed"), ("status", "smoke", "status"),
])
def test_ineligible_candidates(bench, catalogue, field, value, message):
    catalogue["candidates"][1][field] = value
    assert_invalid(bench, catalogue, message)


@pytest.mark.parametrize("field,value", [("protocol", "B"), ("fold", 1), ("fold", False),
                                        ("seed", 43), ("role", "val"), ("role", "test")])
def test_official_training_contract_required(bench, catalogue, field, value):
    catalogue[field] = value
    assert_invalid(bench, catalogue, field)


def test_duplicate_candidates(bench, catalogue):
    catalogue["candidates"][2] = copy.deepcopy(catalogue["candidates"][0])
    assert_invalid(bench, catalogue, "candidate")


@pytest.mark.parametrize("field", ["directory", "revision", "archive_sha256", "inventory_sha256"])
def test_swapped_source_binding(bench, catalogue, field):
    catalogue["candidates"][1]["scientific_source"][field] = catalogue["candidates"][0]["scientific_source"][field]
    assert_invalid(bench, catalogue, "source")


@pytest.mark.parametrize("field", ["source_sha256", "config_sha256", "weights_sha256",
                                   "records_sha256", "split_sha256"])
def test_original_identities_cannot_be_rewritten(bench, catalogue, field):
    catalogue["candidates"][0]["identities"][field] = digest("rewritten")
    assert_invalid(bench, catalogue, "identit")


@pytest.mark.parametrize("mutation", ["foreign", "reverse", "later", "duplicate"])
def test_exact_first_200_train_ids(bench, catalogue, mutation):
    if mutation == "foreign":
        catalogue["image_ids"][0] = "id2001"
    elif mutation == "reverse":
        catalogue["image_ids"].reverse()
    elif mutation == "later":
        catalogue["image_ids"] = [f"id{index:04d}" for index in range(1, 201)]
    else:
        catalogue["image_ids"][1] = catalogue["image_ids"][0]
    assert_invalid(bench, catalogue, "first 200")


@pytest.mark.parametrize("field", ["label", "design_id", "component_id", "path", "attack_family"])
def test_complete_training_membership_must_match_records(bench, catalogue, field):
    candidate = catalogue["candidates"][0]
    complete(candidate)
    path = Path(candidate["train_json"]["path"])
    train = json.loads(path.read_text())
    train["membership"]["train"][400][field] = 1 if field == "label" else "rewritten"
    candidate["train_json"] = write_json(path, train)
    assert_invalid(bench, catalogue, "membership")


def test_modified_metadata_authority_rejected(bench, catalogue):
    Path(catalogue["candidates"][0]["records"]["path"]).write_text("modified")
    assert_invalid(bench, catalogue, "SHA256")


def test_missing_metadata_is_not_waiting(bench, catalogue):
    Path(catalogue["candidates"][0]["split"]["path"]).unlink()
    assert_invalid(bench, catalogue, "split")


@pytest.mark.parametrize("mutation", ["config", "phase", "roles", "selected", "epochs", "count"])
def test_complete_ledger_metadata_binding(bench, catalogue, mutation):
    candidate = catalogue["candidates"][0]
    complete(candidate)
    path = Path(candidate["train_json"]["path"])
    train = json.loads(path.read_text())
    if mutation == "config":
        train["config"]["track"] = "D"
    elif mutation == "phase":
        train["phase"] = "overfit"
    elif mutation == "roles":
        train["opened_roles"].append("test")
    elif mutation == "selected":
        train["selected_checkpoint_sha256"] = digest("other")
    elif mutation == "epochs":
        train["epochs_completed"] = 99
    else:
        train["n_images"]["train"] = 16
    candidate["train_json"] = write_json(path, train)
    assert_invalid(bench, catalogue, "ledger")


@pytest.mark.parametrize("missing", ["train_json", "selected_checkpoint", "selected_evidence", "completion_review"])
def test_missing_completion_authority_blocks_ready(bench, catalogue, missing):
    for candidate in catalogue["candidates"]:
        complete(candidate)
    catalogue["candidates"][1][missing] = None
    result = bench.prepare_bindings(catalogue)
    assert result["status"] == "waiting-for-completion", result
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["scope", "approval", "checkpoint", "evidence", "train"])
def test_review_must_bind_final_selected_evidence(bench, catalogue, mutation):
    candidate = catalogue["candidates"][0]
    complete(candidate)
    review = candidate["completion_review"]
    if mutation == "scope":
        review["scope"] = "pretraining"
    elif mutation == "approval":
        review["approved"] = False
    elif mutation == "checkpoint":
        review["bindings"]["selected_checkpoint_sha256"] = digest("other")
    elif mutation == "evidence":
        review["bindings"]["selected_evidence_sha256"].pop("gate.json")
    else:
        review["bindings"]["train_json_sha256"] = digest("other")
    assert_invalid(bench, catalogue, "review")


def test_historical_report_absence_is_legitimate(bench, catalogue):
    assert not (Path(catalogue["candidates"][0]["scientific_source"]["directory"]) /
                "scripts/report_anchor_pilots.py").exists()
    assert bench.prepare_bindings(catalogue)["status"] == "waiting-for-completion"


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "escape", "symlink"])
def test_measurement_tool_authority_required(bench, catalogue, tmp_path, mutation):
    tools = catalogue["measurement_tools"]
    path = Path(tools["directory"]) / "report_anchor_pilots.py"
    if mutation == "missing":
        path.unlink()
    elif mutation == "corrupt":
        path.write_text("changed")
    elif mutation == "escape":
        tools["files"]["../outside.py"] = digest("outside")
    else:
        target = tmp_path / "outside.py"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    assert_invalid(bench, catalogue, "measurement")


def test_staged_metadata_fallback_only_to_bound_parent(bench, catalogue):
    candidate = catalogue["candidates"][0]
    complete(candidate)
    path = Path(candidate["train_json"]["path"])
    train = json.loads(path.read_text())
    train["paths"]["records"] = str(path.parent / "gone/records.csv")
    train["paths"]["split"] = str(path.parent / "gone/split.csv")
    candidate["train_json"] = write_json(path, train)
    candidate["completion_review"]["bindings"]["train_json_sha256"] = candidate["train_json"]["sha256"]
    assert bench.prepare_bindings(catalogue)["status"] == "waiting-for-completion"
    catalogue["candidates"][0]["records"]["path"] = str(path.parent / "unbound.csv")
    assert_invalid(bench, catalogue, "records")


def test_explicit_hash_checked_metadata_relocation(bench, catalogue, tmp_path):
    candidate = catalogue["candidates"][0]
    original = candidate["records"]["path"]
    relocated = tmp_path / "relocated-records.csv"
    relocated.write_bytes(Path(original).read_bytes())
    candidate["records"]["path"] = str(relocated)
    candidate["relocations"] = {"records": {"from": original, "to": str(relocated),
                                            "sha256": candidate["records"]["sha256"]}}
    assert bench.prepare_bindings(catalogue)["status"] == "waiting-for-completion"
    candidate["relocations"]["records"]["sha256"] = digest("wrong")
    assert_invalid(bench, catalogue, "relocation")


@pytest.mark.parametrize("mutation", ["protocol", "fold", "join", "counts", "duplicate", "path"])
def test_full_official_csv_metadata_is_validated(bench, catalogue, mutation):
    kind = "records" if mutation == "path" else "split"
    path = Path(catalogue["candidates"][0][kind]["path"])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if mutation == "protocol":
        rows[0]["protocol"] = "B"
    elif mutation == "fold":
        rows[0]["ref_group"] = "it1"
    elif mutation == "join":
        rows[0]["component_id"] = "foreign"
    elif mutation == "counts":
        rows[0]["role"] = "test"
    elif mutation == "duplicate":
        rows[1]["image_id"] = rows[0]["image_id"]
    else:
        rows[0]["path"] = "../escaped.jpg"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for candidate in catalogue["candidates"]:
        candidate[kind] = authority(path)
        candidate["identities"][f"{kind}_sha256"] = candidate[kind]["sha256"]
        parent_path = Path(candidate["parent_manifest"]["path"])
        parent = json.loads(parent_path.read_text())
        parent["runtime_identities"] = candidate["identities"]
        parent["provenance"][f"{kind}_sha256"] = candidate[kind]["sha256"]
        candidate["parent_manifest"] = write_json(parent_path, parent)
    assert_invalid(bench, catalogue, "metadata")


def cli(catalogue, tmp_path, phase="prepare", output=None, expected_hash=None):
    path = tmp_path / "catalogue.json"
    binding = write_json(path, catalogue)
    destination = output or tmp_path / "readiness"
    result = subprocess.run([sys.executable, str(SCRIPT), "--phase", phase,
                             "--candidates", str(path), "--candidates-sha256",
                             expected_hash or binding["sha256"], "--run-dir", str(destination)],
                            capture_output=True, text=True)
    return result, destination


@pytest.mark.parametrize("state,exit_code", [("pending", 2), ("complete", 0), ("failed", 1)])
def test_cli_readiness_exit_codes(bench, catalogue, tmp_path, state, exit_code):
    if state == "complete":
        for candidate in catalogue["candidates"]:
            complete(candidate)
    elif state == "failed":
        catalogue["candidates"][1]["status"] = "failed"
    result, output = cli(catalogue, tmp_path)
    assert result.returncode == exit_code, result.stderr
    payload = json.loads((output / "readiness.json").read_text())
    assert payload["comparable"] is False and payload["execution_authorized"] is False
    assert payload["catalogue_sha256"] == authority(tmp_path / "catalogue.json")["sha256"]


def test_cli_modified_external_catalogue_authority(bench, catalogue, tmp_path):
    result, output = cli(catalogue, tmp_path, expected_hash=digest("original"))
    assert result.returncode == 1
    assert "SHA256" in result.stderr
    assert not output.exists()


def test_measure_phase_requires_cpu_allocation_even_when_metadata_ready(bench, catalogue, tmp_path):
    for candidate in catalogue["candidates"]:
        complete(candidate)
    result, output = cli(catalogue, tmp_path, phase="measure")
    assert result.returncode == 1
    assert "CPU" in result.stderr
    assert not output.exists()


def test_duplicate_json_keys_rejected_before_outputs(bench, catalogue, tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"first","schema_version":"second"}')
    result = subprocess.run([sys.executable, str(SCRIPT), "--phase", "prepare", "--candidates", str(path),
                             "--candidates-sha256", authority(path)["sha256"],
                             "--run-dir", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 1 and "duplicate" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["source", "data", "run", "metadata", "measurement", "ancestor", "symlink", "existing"])
def test_cli_output_guards_actual_normalized_path(bench, catalogue, tmp_path, kind):
    candidate = catalogue["candidates"][0]
    if kind == "source":
        protected = Path(candidate["scientific_source"]["directory"])
    elif kind == "data":
        protected = Path(candidate["raw_root"])
        protected.mkdir()
    elif kind == "run":
        protected = Path(candidate["run_dir"])
    elif kind == "metadata":
        protected = Path(candidate["records"]["path"]).parent
    elif kind == "measurement":
        protected = Path(catalogue["measurement_tools"]["directory"])
    elif kind == "ancestor":
        protected = tmp_path
    elif kind == "existing":
        protected = tmp_path / "existing"
        protected.mkdir()
    else:
        protected = tmp_path / "link"
        protected.symlink_to(Path(candidate["scientific_source"]["directory"]), target_is_directory=True)
    if kind in ("ancestor", "existing"):
        output = protected
    elif kind == "symlink":
        output = protected / "../output"
    else:
        output = tmp_path / "outside" / ".." / protected.relative_to(tmp_path) / "new-output"
    result, unused = cli(catalogue, tmp_path, output=output)
    assert result.returncode == 1, result.stderr
    assert not (protected / "new-output").exists()


def test_new_output_is_exclusive(bench, catalogue, tmp_path):
    result, output = cli(catalogue, tmp_path)
    assert result.returncode == 2
    original = (output / "readiness.json").read_bytes()
    result, unused = cli(catalogue, tmp_path, output=output)
    assert result.returncode == 1
    assert (output / "readiness.json").read_bytes() == original


def test_stdlib_only_import_in_fresh_interpreter(bench):
    code = f"""
import importlib.abc
import importlib.util
import sys
class BlockScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'cv2', 'numpy', 'refer_or_clear'}}:
            raise AssertionError('forbidden import ' + fullname)
sys.meta_path.insert(0, BlockScientific())
spec = importlib.util.spec_from_file_location('candidate_preflight', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.prepare_bindings({{}})['status'] == 'not-comparable'
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pinned_contract_preserves_old_semantics_without_importing_old_runner(bench):
    tree = ast.parse((ROOT / "scripts/bench_anchor_pilots.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "CONTRACT" for target in node.targets))
    expected = ast.literal_eval(assignment.value)
    expected.update(name="anchor_candidate_cpu_v1", track_order=list(TRACKS))
    assert bench.CONTRACT == expected


def test_measurement_tooling_is_separately_frozen(bench, catalogue):
    result = bench.prepare_bindings(catalogue)
    assert result["measurement_tools"] == catalogue["measurement_tools"]
    assert result["measurement_tools"] is not catalogue["measurement_tools"]


@pytest.mark.parametrize("field", ["records_sha256", "split_sha256", "weights_sha256"])
def test_parent_provenance_matches_runtime_metadata_identities(bench, catalogue, field):
    candidate = catalogue["candidates"][0]
    path = Path(candidate["parent_manifest"]["path"])
    parent = json.loads(path.read_text())
    parent["provenance"][field] = digest("other")
    candidate["parent_manifest"] = write_json(path, parent)
    assert_invalid(bench, catalogue, "provenance")


def test_original_staged_ledger_inputs_are_protected(bench, catalogue, tmp_path):
    candidate = catalogue["candidates"][0]
    complete(candidate)
    stage = tmp_path / "original-stage"
    stage.mkdir()
    path = Path(candidate["train_json"]["path"])
    train = json.loads(path.read_text())
    train["paths"]["raw_root"] = str(stage)
    candidate["train_json"] = write_json(path, train)
    candidate["completion_review"]["bindings"]["train_json_sha256"] = candidate["train_json"]["sha256"]
    result, unused = cli(catalogue, tmp_path, output=stage / "new-report")
    assert result.returncode == 1
    assert not (stage / "new-report").exists()


def test_prepare_refuses_checkpoint_bytes_mislabeled_as_a_tool(bench, catalogue, tmp_path, monkeypatch):
    path = Path(catalogue["measurement_tools"]["directory"]) / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    catalogue["measurement_tools"]["files"][path.name] = authority(path)["sha256"]
    original_read = Path.read_bytes

    def no_checkpoint_read(requested):
        if requested.suffix == ".pt":
            pytest.fail("prepare must not read checkpoint bytes")
        return original_read(requested)

    monkeypatch.setattr(Path, "read_bytes", no_checkpoint_read)
    assert_invalid(bench, catalogue, "measurement")


def test_bad_available_authority_overrides_waiting(bench, catalogue):
    candidate = catalogue["candidates"][0]
    complete(candidate)
    candidate["status"] = "pending"
    Path(candidate["selected_evidence"]["gate.json"]["path"]).write_text("corrupted")
    assert_invalid(bench, catalogue, "SHA256")


@pytest.mark.parametrize("payload", [None, [], {}, {"schema_version": "wrong"}])
def test_malformed_catalogue_is_noncomparable(bench, payload):
    result = bench.prepare_bindings(payload)
    assert result["status"] == "not-comparable"
    assert result["errors"] and result["execution_authorized"] is False


def test_nonfinite_catalogue_field_is_malformed(bench, catalogue):
    catalogue["candidates"][0]["extra"] = float("inf")
    assert_invalid(bench, catalogue, "JSON")


def test_json_number_overflow_is_rejected_before_outputs(bench, tmp_path):
    path = tmp_path / "overflow.json"
    path.write_text('{"extra":1e999}')
    result = subprocess.run([sys.executable, str(SCRIPT), "--phase", "prepare", "--candidates", str(path),
                             "--candidates-sha256", authority(path)["sha256"],
                             "--run-dir", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 1
    assert "JSON" in result.stderr
    assert not (tmp_path / "output").exists()


def test_multiple_invalid_candidates_report_each_binding_error(bench, catalogue):
    for candidate in catalogue["candidates"]:
        candidate["architecture_id"] = "rewritten"
    result = bench.prepare_bindings(catalogue)
    assert result["status"] == "not-comparable"
    assert all(any(error.startswith(track + ":") for error in result["errors"]) for track in TRACKS)


def refresh_runtime_ledger(candidate, train):
    candidate["train_json"] = write_json(Path(candidate["train_json"]["path"]), train)
    candidate["completion_review"]["bindings"].update(
        train_json_sha256=candidate["train_json"]["sha256"],
        selected_checkpoint_sha256=candidate["selected_checkpoint"]["sha256"])


@pytest.fixture
def runtime_catalogue(bench, catalogue):
    assert WORKER.is_file(), "candidate scientific worker is not implemented"
    tools = Path(catalogue["measurement_tools"]["directory"])
    fake_weights = {}
    for track, candidate in zip(TRACKS, catalogue["candidates"]):
        weights = Path(candidate["weights"]["path"])
        weights.write_bytes(b"synthetic R weights" if track == "R" else b"synthetic IA weights")
        fake_weights[track] = authority(weights)["sha256"]
    coordinator_code = SCRIPT.read_text()
    for track in TRACKS:
        coordinator_code = coordinator_code.replace(WEIGHTS[track], fake_weights[track])
    (tools / SCRIPT.name).write_text(coordinator_code)
    (tools / WORKER.name).write_bytes(WORKER.read_bytes())
    bench.OFFLINE_WEIGHTS = fake_weights
    for candidate in catalogue["candidates"]:
        source = Path(candidate["scientific_source"]["directory"])
        package = source / "src/refer_or_clear"
        (package / "data").mkdir(parents=True)
        (source / "scripts").mkdir()
        (package / "__init__.py").write_text(f"SOURCE_TAG = {candidate['scientific_source']['revision']!r}\n")
        (package / "data/__init__.py").write_text("")
        (source / "scripts/check_anchor_pilots.py").write_text("SCIENTIFIC_CHECK = True\n")
        (package / "anchor_contract.py").write_text('''
from types import SimpleNamespace
from . import SOURCE_TAG
def contract_for(track):
    names = {'R': 'anchor_rgb_r50_v1', 'D_running_pilot': 'anchor_direct_running_ia18_pilot_v1',
             'N': 'anchor_absent_ia18_v1'}
    return SimpleNamespace(architecture_id=names[track], source_tag=SOURCE_TAG, bn_policy='running',
                           input_shape=(3, 299, 299) if track == 'R' else (4, 512, 800),
                           data_contract='raw_rgb_299_v1' if track == 'R' else 'legacy_geometry_diagnostic')
''')
        (package / "pilots.py").write_text('''
import hashlib
import json
from pathlib import Path
from . import SOURCE_TAG
def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def source_sha256():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / 'src').rglob('*.py')) + [root / 'scripts/check_anchor_pilots.py']
    return json_sha256({str(path.relative_to(root)): file_sha256(path) for path in files})
def read_pilot_checkpoint(path, config, identities):
    saved = json.loads(Path(path).read_text())
    if saved['config'] != config or saved['identities'] != identities:
        raise ValueError('checkpoint config/identity mismatch')
    if saved['architecture_id'] != config['architecture_id'] or saved['phase'] != 'train':
        raise ValueError('checkpoint architecture/phase mismatch')
    return saved
def build_initialized_model(*args, **kwargs):
    raise AssertionError('model construction forbidden during inspection/preparation ' + SOURCE_TAG)
''')
        (package / "data/query.py").write_text('''
import csv
class Rows:
    def __init__(self, rows):
        self.rows = rows
        self.iloc = self
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, selection):
        return Rows(self.rows[selection])
    def copy(self):
        return Rows([dict(row) for row in self.rows])
    def to_dict(self, orientation):
        assert orientation == 'records'
        return self.rows
    def __setitem__(self, key, value):
        for row in self.rows:
            row[key] = value
def load_role_rows(records_csv, split_csv, role):
    assert role == 'train', 'never load validation/test for timing'
    with open(records_csv, newline='') as stream:
        records = list(csv.DictReader(stream))
    with open(split_csv, newline='') as stream:
        ids = {row['image_id'] for row in csv.DictReader(stream) if row['role'] == role}
    return Rows([{**row, 'label': int(row['label']), 'role': role} for row in records if row['image_id'] in ids])
class AnchorQueryDataset:
    def __init__(self, rows, raw_root, legacy_root, track, train, augment):
        assert train is False and augment == {'enabled': False}
        assert all(row['role'] == 'val' for row in rows.to_dict('records'))
    def __getitem__(self, index):
        raise AssertionError('no query evaluation in Task 2')
''')
        source_files = [{"path": str(path.relative_to(source)), **{key: value for key, value in authority(path).items()
                                                                  if key != "path"}, "size": path.stat().st_size}
                        for path in sorted(source.rglob("*.py"))]
        identities = candidate["identities"]
        identities["weights_sha256"] = fake_weights[candidate["track"]]
        identities["source_sha256"] = digest({item["path"]: item["sha256"] for item in source_files})
        candidate["weights"]["sha256"] = identities["weights_sha256"]
        candidate["scientific_source"]["inventory_sha256"] = digest(source_files)
        parent_path = Path(candidate["parent_manifest"]["path"])
        parent = json.loads(parent_path.read_text())
        parent.update(runtime_identities=identities, source_files=source_files)
        parent["provenance"].update(source_sha256=digest(source_files), weights_sha256=identities["weights_sha256"])
        candidate["parent_manifest"] = write_json(parent_path, parent)
        complete(candidate)
        checkpoint = Path(candidate["selected_checkpoint"]["path"])
        checkpoint.parent.mkdir()
        saved = {"architecture_id": candidate["architecture_id"], "phase": "train", "epoch": 55,
                 "selection_key": [0.9, 0.95], "config": parent["effective_config"], "identities": identities}
        candidate["selected_checkpoint"] = write_json(checkpoint, saved)
        train = json.loads(Path(candidate["train_json"]["path"]).read_text())
        train.update(selected_checkpoint_sha256=candidate["selected_checkpoint"]["sha256"],
                     n_active_parameters=12, n_total_parameters=15, initialization={"passed": True})
        refresh_runtime_ledger(candidate, train)
    (tools / "report_anchor_pilots.py").write_text('''
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from refer_or_clear.pilots import read_pilot_checkpoint, file_sha256
from refer_or_clear.anchor_contract import contract_for
def inspect_run(run_dir, track):
    run_dir = Path(run_dir)
    train = json.loads((run_dir / 'train.json').read_text())
    saved = read_pilot_checkpoint(run_dir / 'checkpoints/best.pt', train['config'], train['identities'])
    if saved['epoch'] != train['selected_epoch'] or saved['selection_key'] != train['selection_key']:
        raise ValueError('checkpoint selected epoch/key mismatch')
    contract = contract_for(track)
    if contract.architecture_id != train['architecture_id']:
        raise ValueError('unknown architecture')
    return {'track': track, 'run_dir': str(run_dir), 'status': 'complete', 'eligible_candidate': True,
            'acceptance_targets_met': False, 'selected_epoch': saved['epoch'], 'identities': train['identities'],
            'data_contract': contract.data_contract, 'n_active_parameters': 12, 'n_total_parameters': 15,
            'checkpoint_bytes': (run_dir / 'checkpoints/best.pt').stat().st_size,
            'artifact_sha256': {name: file_sha256(run_dir / name) for name in
                ('train.json', 'val_summary.json', 'val_predictions.csv', 'log.csv', 'gate.json', 'replay.json')}}
''')
    (tools / "bench_anchor_pilots.py").write_text('''
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from refer_or_clear.anchor_contract import contract_for
from refer_or_clear.pilots import file_sha256, json_sha256, build_initialized_model, read_pilot_checkpoint
from refer_or_clear.data.query import load_role_rows, AnchorQueryDataset
def configure_threads():
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
def host_info():
    return {'hostname': 'cpu01', 'boot_id': 'boot01', 'cpu_model': 'synthetic CPU', 'affinity': [0, 1, 2, 3],
            'allocated_cpus': 4, 'job_id': '123', 'node': 'cpu01', 'os': 'synthetic',
            'versions': {'python': 'synthetic'}, 'environment_prefix': 'synthetic', 'python_executable': sys.executable}
def validate_host(host):
    assert host['allocated_cpus'] >= 4 and len(host['affinity']) >= 4
def resolve_weights(run_dir, info):
    return Path(info['paths']['weights'])
def resolve_metadata(run_dir, info, key):
    return Path(info['paths'][key])
def runtime_train_rows(info):
    assert len(info['membership']['train']) == 1778
    return sorted(info['membership']['train'], key=lambda row: row['image_id'])[:200]
def source_paths(row, track, raw_root, legacy_root):
    return [Path(raw_root) / row['path']] if track == 'R' else [
        Path(legacy_root) / 'canvases' / (row['image_id'] + '.png'), Path(legacy_root) / 'zones/design.png']
def source_artifacts(row, track, raw_root, legacy_root):
    return [{'path': str(path), 'sha256': file_sha256(path),
             'shape': [299, 299, 3] if track == 'R' else [512, 800] if index else [512, 800, 3]}
            for index, path in enumerate(source_paths(row, track, raw_root, legacy_root))]
''')
    misleading = tools / "src/refer_or_clear"
    misleading.mkdir(parents=True)
    (misleading / "__init__.py").write_text("raise AssertionError('measurement source must not replace scientific source')\n")
    catalogue["measurement_tools"]["files"] = {name: authority(tools / name)["sha256"] for name in
                                               (SCRIPT.name, WORKER.name, "bench_anchor_pilots.py", "report_anchor_pilots.py")}
    return catalogue


def worker_request(runtime_catalogue, tmp_path, operation="inspect", track="R", inspections=None):
    catalogue_binding = write_json(tmp_path / "runtime-catalogue.json", runtime_catalogue)
    request = {"schema_version": "anchor_candidate_worker_request_v1", "operation": operation,
               "catalogue": catalogue_binding, "track": track, "inspections": inspections}
    return write_json(tmp_path / f"request-{operation}-{track}.json", request)


def execute_worker(runtime_catalogue, request, output, environment=None):
    import os
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SLURM_JOB_ID="123", SLURMD_NODENAME="cpu01",
               SLURM_JOB_PARTITION="cpu", SLURM_CPUS_PER_TASK="4", CUDA_VISIBLE_DEVICES="", PYTHONPATH="/poisoned")
    env.update(environment or {})
    script = Path(runtime_catalogue["measurement_tools"]["directory"]) / WORKER.name
    return subprocess.run([sys.executable, "-I", str(script), "--operation",
                           json.loads(Path(request["path"]).read_text())["operation"],
                           "--request", request["path"], "--request-sha256", request["sha256"],
                           "--out", str(output)], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("track", TRACKS)
def test_worker_inspection_uses_own_scientific_package_without_historical_report(runtime_catalogue, tmp_path, track):
    request = worker_request(runtime_catalogue, tmp_path, track=track)
    output = tmp_path / f"inspection-{track}.json"
    result = execute_worker(runtime_catalogue, request, output)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    candidate = next(item for item in runtime_catalogue["candidates"] if item["track"] == track)
    source = Path(candidate["scientific_source"]["directory"])
    assert payload["status"] == "inspected" and payload["operation"] == "inspect"
    assert payload["inspection"]["acceptance_targets_met"] is False
    assert payload["identities"] == candidate["identities"]
    assert all(Path(origin).is_relative_to(source / "src") for origin in payload["scientific_module_origins"].values())
    assert not (source / "scripts/report_anchor_pilots.py").exists()
    assert payload["comparable"] is False and payload["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["source", "extra_source", "weights", "checkpoint", "tool", "missing_tool",
                                       "review", "architecture", "request_hash", "origin", "epoch", "key", "identity", "config"])
def test_worker_rejects_mutated_scientific_authorities_before_preparation(runtime_catalogue, tmp_path, mutation):
    candidate = runtime_catalogue["candidates"][0]
    tools = Path(runtime_catalogue["measurement_tools"]["directory"])
    if mutation == "source":
        (Path(candidate["scientific_source"]["directory"]) / "src/refer_or_clear/pilots.py").write_text("changed")
    elif mutation == "extra_source":
        (Path(candidate["scientific_source"]["directory"]) / "src/unbound.py").write_text("unbound")
    elif mutation == "weights":
        Path(candidate["weights"]["path"]).write_bytes(b"changed")
    elif mutation == "checkpoint":
        Path(candidate["selected_checkpoint"]["path"]).write_text("changed")
    elif mutation == "tool":
        (tools / "report_anchor_pilots.py").write_text("changed")
    elif mutation == "missing_tool":
        (tools / "report_anchor_pilots.py").unlink()
    elif mutation == "review":
        candidate["completion_review"] = None
    elif mutation == "architecture":
        candidate["architecture_id"] = "unknown"
    elif mutation == "origin":
        path = tools / "report_anchor_pilots.py"
        path.write_text(path.read_text() + "\nimport refer_or_clear\nrefer_or_clear.__path__.append('/foreign')\n")
        runtime_catalogue["measurement_tools"]["files"][path.name] = authority(path)["sha256"]
    elif mutation in ("epoch", "key", "identity", "config"):
        path = Path(candidate["selected_checkpoint"]["path"])
        saved = json.loads(path.read_text())
        if mutation == "epoch":
            saved["epoch"] = 100
        elif mutation == "key":
            saved["selection_key"] = [1.0, 1.0]
        elif mutation == "identity":
            saved["identities"]["source_sha256"] = digest("foreign")
        else:
            saved["config"]["batch_size"] = 1
        candidate["selected_checkpoint"] = write_json(path, saved)
        train = json.loads(Path(candidate["train_json"]["path"]).read_text())
        train["selected_checkpoint_sha256"] = candidate["selected_checkpoint"]["sha256"]
        refresh_runtime_ledger(candidate, train)
    request = worker_request(runtime_catalogue, tmp_path)
    if mutation == "request_hash":
        request["sha256"] = digest("wrong")
    output = tmp_path / "inspection.json"
    result = execute_worker(runtime_catalogue, request, output)
    assert result.returncode == 1, result.stdout
    assert result.stderr and not output.exists()


def test_prepare_requires_all_three_inspections_before_any_input_access(runtime_catalogue, tmp_path):
    request = worker_request(runtime_catalogue, tmp_path, operation="prepare")
    result = execute_worker(runtime_catalogue, request, tmp_path / "prepared.json")
    assert result.returncode == 1 and "inspection" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def all_inspections(runtime_catalogue, tmp_path):
    results = {}
    for track in TRACKS:
        request = worker_request(runtime_catalogue, tmp_path, track=track)
        output = tmp_path / f"inspected-{track}.json"
        result = execute_worker(runtime_catalogue, request, output)
        assert result.returncode == 0, result.stderr
        results[track] = authority(output)
    return results


def synthetic_inputs(runtime_catalogue):
    for candidate in runtime_catalogue["candidates"]:
        if candidate["track"] == "R":
            for image_id in runtime_catalogue["image_ids"]:
                path = Path(candidate["raw_root"]) / f"raw/{image_id}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic raw bytes, not an image")
        else:
            for image_id in runtime_catalogue["image_ids"]:
                path = Path(candidate["legacy_root"]) / f"canvases/{image_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic canvas bytes, not an image")
            path = Path(candidate["legacy_root"]) / "zones/design.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"synthetic zone bytes, not an image")


@pytest.mark.parametrize("track", TRACKS)
def test_worker_prepares_only_frozen_training_ids_and_query_only_view(runtime_catalogue, tmp_path, track):
    inspections = all_inspections(runtime_catalogue, tmp_path)
    synthetic_inputs(runtime_catalogue)
    request = worker_request(runtime_catalogue, tmp_path, operation="prepare", track=track, inspections=inspections)
    output = tmp_path / "prepared.json"
    result = execute_worker(runtime_catalogue, request, output)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "prepared" and payload["image_ids"] == runtime_catalogue["image_ids"]
    assert payload["membership_role"] == "train" and payload["query_view_role"] == "val"
    assert payload["input_shape"] == ([1, 3, 299, 299] if track == "R" else [1, 4, 512, 800])
    assert len(payload["sources"]) == 200
    assert all(len(row["artifacts"]) == (1 if track == "R" else 2) for row in payload["sources"])
    assert "durations" not in payload and payload["comparable"] is False


def test_prepare_rejects_cross_candidate_inspection_mutation_before_inputs(runtime_catalogue, tmp_path):
    inspections = all_inspections(runtime_catalogue, tmp_path)
    Path(runtime_catalogue["candidates"][1]["weights"]["path"]).write_bytes(b"changed")
    request = worker_request(runtime_catalogue, tmp_path, operation="prepare", inspections=inspections)
    result = execute_worker(runtime_catalogue, request, tmp_path / "prepared.json")
    assert result.returncode == 1 and "SHA256" in result.stderr


def test_worker_measure_requires_frozen_manifest_and_inspections(runtime_catalogue, tmp_path):
    request = worker_request(runtime_catalogue, tmp_path, operation="measure")
    result = execute_worker(runtime_catalogue, request, tmp_path / "measurement.json")
    assert result.returncode == 1 and "manifest" in result.stderr


def test_coordinator_pending_metadata_invokes_no_worker(bench, catalogue, tmp_path):
    assert callable(getattr(bench, "prepare_candidate_runtime", None)), "Task 2 coordinator not implemented"
    binding = write_json(tmp_path / "catalogue.json", catalogue)

    def forbidden_runner(*args):
        pytest.fail("pending candidates must not start a worker")

    result = bench.prepare_candidate_runtime(binding, tmp_path / "runtime", worker_runner=forbidden_runner)
    assert result["status"] == "waiting-for-completion" and result["comparable"] is False
    assert not (tmp_path / "runtime").exists()


def test_shared_manifest_is_frozen_only_after_all_inspections(bench, runtime_catalogue, tmp_path, monkeypatch):
    synthetic_inputs(runtime_catalogue)
    binding = write_json(tmp_path / "catalogue.json", runtime_catalogue)
    events = []
    monkeypatch.setattr(bench, "validate_allocation", lambda: None)

    def runner(operation, request, output):
        events.append((operation, json.loads(Path(request["path"]).read_text())["track"]))
        result = execute_worker(runtime_catalogue, request, output)
        assert result.returncode == 0, result.stderr

    result = bench.prepare_candidate_runtime(binding, tmp_path / "runtime", worker_runner=runner)
    assert events == [(operation, track) for operation in ("inspect", "prepare") for track in TRACKS]
    assert result["status"] == "prepared" and result["comparable"] is False
    path = tmp_path / "runtime/runtime-manifest.json"
    manifest = json.loads(path.read_text())
    declared = manifest.pop("manifest_sha256")
    assert declared == digest(manifest)
    assert manifest["schema_version"] == "anchor_candidate_cpu_v1"
    assert manifest["image_ids"] == runtime_catalogue["image_ids"]
    assert list(manifest["contract"]["track_order"]) == list(TRACKS)
    assert manifest["tracks"]["R"]["identities"]["source_sha256"] != manifest["tracks"]["D_running_pilot"]["identities"]["source_sha256"]
    assert manifest["full_source_to_decision_latency"] == "not_measured"
    assert manifest["measurement_tools"] == runtime_catalogue["measurement_tools"]
    original = path.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        bench.freeze_candidate_manifest(path, manifest)
    assert path.read_bytes() == original


def test_failed_later_inspection_never_starts_preparation(bench, runtime_catalogue, tmp_path, monkeypatch):
    binding = write_json(tmp_path / "catalogue.json", runtime_catalogue)
    monkeypatch.setattr(bench, "validate_allocation", lambda: None)
    events = []

    def runner(operation, request, output):
        track = json.loads(Path(request["path"]).read_text())["track"]
        events.append((operation, track))
        if track == "D_running_pilot":
            raise ValueError("synthetic failed inspection")
        result = execute_worker(runtime_catalogue, request, output)
        assert result.returncode == 0, result.stderr

    result = bench.prepare_candidate_runtime(binding, tmp_path / "runtime", worker_runner=runner)
    assert result["status"] == "not-comparable" and result["comparable"] is False
    assert events == [("inspect", "R"), ("inspect", "D_running_pilot")]
    assert not (tmp_path / "runtime/runtime-manifest.json").exists()


def test_worker_rejects_unbound_cached_scientific_code(runtime_catalogue, tmp_path):
    source = Path(runtime_catalogue["candidates"][0]["scientific_source"]["directory"])
    cached = source / "src/refer_or_clear/__pycache__/pilots.cpython-310.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"unbound cached executable code")
    request = worker_request(runtime_catalogue, tmp_path)
    result = execute_worker(runtime_catalogue, request, tmp_path / "inspection.json")
    assert result.returncode == 1 and "cached" in result.stderr


@pytest.mark.parametrize("environment", [
    {"SLURM_JOB_PARTITION": "gpu"}, {"SLURM_JOB_ID": ""}, {"SLURMD_NODENAME": "login01"},
    {"SLURM_CPUS_PER_TASK": "3"}, {"CUDA_VISIBLE_DEVICES": "0"},
])
def test_worker_allocation_gate_precedes_scientific_import(runtime_catalogue, tmp_path, environment):
    request = worker_request(runtime_catalogue, tmp_path)
    result = execute_worker(runtime_catalogue, request, tmp_path / "inspection.json", environment)
    assert result.returncode == 1 and "import" not in result.stderr


@pytest.mark.parametrize("mutation", ["stale_input", "host", "shape", "checkpoint", "tooling", "role", "parameters", "mapping"])
def test_preparation_manifest_rejects_stale_or_rewritten_worker_evidence(bench, runtime_catalogue, tmp_path, monkeypatch, mutation):
    synthetic_inputs(runtime_catalogue)
    binding = write_json(tmp_path / "catalogue.json", runtime_catalogue)
    monkeypatch.setattr(bench, "validate_allocation", lambda: None)

    def runner(operation, request, output):
        result = execute_worker(runtime_catalogue, request, output)
        assert result.returncode == 0, result.stderr
        if operation != "prepare":
            return
        payload = json.loads(output.read_text())
        if mutation == "stale_input" and payload["track"] == "N":
            path = Path(runtime_catalogue["candidates"][0]["raw_root"]) / "raw/id0000.jpg"
            path.write_bytes(b"changed synthetic training bytes")
        elif mutation == "host":
            payload["host"]["boot_id"] = "foreign-boot"
        elif mutation == "shape":
            payload["input_shape"] = [1, 4, 1, 1]
        elif mutation == "checkpoint":
            payload["checkpoint_sha256"] = digest("wrong")
        elif mutation == "tooling":
            payload["measurement_tools"] = {}
        elif mutation == "role":
            payload["membership_role"] = "test"
        elif mutation == "parameters":
            payload["n_active_parameters"] = 1
        elif mutation == "mapping":
            payload["sources"][0]["image_id"] = "id2001"
            payload["sources_sha256"] = digest(payload["sources"])
        output.write_text(json.dumps(payload))

    result = bench.prepare_candidate_runtime(binding, tmp_path / "runtime", worker_runner=runner)
    assert result["status"] == "not-comparable", result
    assert result["errors"] and not (tmp_path / "runtime/runtime-manifest.json").exists()


@pytest.mark.parametrize("kind", ["source", "data", "existing", "symlink", "request"])
def test_worker_output_cannot_mutate_inputs(runtime_catalogue, tmp_path, kind):
    candidate = runtime_catalogue["candidates"][0]
    request = worker_request(runtime_catalogue, tmp_path)
    if kind == "source":
        output = tmp_path / "outside/.." / Path(candidate["scientific_source"]["directory"]).name / "new.json"
    elif kind == "data":
        output = Path(candidate["raw_root"]) / "new.json"
        output.parent.mkdir()
    elif kind == "existing":
        output = tmp_path / "existing.json"
        output.write_text("immutable output")
    elif kind == "request":
        output = Path(request["path"])
    else:
        link = tmp_path / "link"
        link.symlink_to(Path(candidate["scientific_source"]["directory"]), target_is_directory=True)
        output = link / "../new.json"
    before = output.read_bytes() if output.is_file() else None
    result = execute_worker(runtime_catalogue, request, output)
    assert result.returncode == 1
    if before is not None:
        assert output.read_bytes() == before


def test_real_worker_runner_clears_pythonpath_and_keeps_interpreter(bench, runtime_catalogue, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("PYTHONPATH", "/untrusted")

    def fake_process(command, **kwargs):
        calls.append((command, kwargs))
        return type("Process", (), {"stdout": "synthetic stdout", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(bench.subprocess, "run", fake_process)
    request = worker_request(runtime_catalogue, tmp_path)
    bench.run_candidate_worker("inspect", request, tmp_path / "worker-result.json")
    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, "-I"]
    assert "PYTHONPATH" not in kwargs["env"]
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert (tmp_path / "worker-result.json.stdout.log").read_text() == "synthetic stdout"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("candidate_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticTensor:
    def __init__(self, shape, value=1, dtype="float32", requires_grad=True):
        self.shape, self.value, self.dtype, self.requires_grad = tuple(shape), value, dtype, requires_grad
        self.device = SimpleNamespace(type="cpu")

    def numel(self):
        return math.prod(self.shape)

    def unsqueeze(self, dimension):
        assert dimension == 0
        return SyntheticTensor((1, *self.shape), self.value, self.dtype)

    def is_floating_point(self):
        return self.dtype.startswith("float")


class SyntheticTorch:
    float32 = "float32"

    def __init__(self):
        self.inference = False
        self.autocast_enabled = False
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                        cudnn=SimpleNamespace(allow_tf32=False))

    @contextmanager
    def inference_mode(self):
        previous = self.inference
        self.inference = True
        try:
            yield
        finally:
            self.inference = previous

    @contextmanager
    def autocast(self, device_type, enabled):
        assert device_type == "cpu" and enabled is False
        yield

    def is_inference_mode_enabled(self):
        return self.inference

    def is_autocast_enabled(self, device_type="cpu"):
        return self.autocast_enabled

    def is_autocast_cpu_enabled(self):
        return self.autocast_enabled

    def get_default_dtype(self):
        return self.float32

    def get_num_threads(self):
        return 4

    def get_num_interop_threads(self):
        return 1


class SyntheticModel:
    def __init__(self, torch_api, events, mutation=None):
        self.training = False
        self.parameters_list = [SyntheticTensor((12,)), SyntheticTensor((3,), requires_grad=False)]
        self.buffer = SyntheticTensor((1,))
        self.torch_api, self.events, self.mutation = torch_api, events, mutation

    def parameters(self):
        return iter(self.parameters_list)

    def named_parameters(self):
        return iter(zip(("active", "frozen"), self.parameters_list))

    def named_buffers(self):
        return iter((("buffer", self.buffer),))

    def state_dict(self):
        return {**dict(self.named_parameters()), **dict(self.named_buffers())}

    def modules(self):
        return iter((self,))

    def __call__(self, tensor):
        assert self.torch_api.inference
        self.events.append("network")
        if self.mutation == "parameters":
            self.parameters_list[0].value += 1
        if self.mutation == "buffers":
            self.buffer.value += 1
        if self.mutation == "mode":
            self.training = True
        events = self.events

        class Score:
            def reshape(self, *shape):
                return self

            def __getitem__(self, index):
                return self

            def item(self):
                events.append("materialize")
                return 0.0

        return {"attack_logit": Score()}


def linear_stats(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {"median": statistics.median(ordered),
            "p95": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)}


@pytest.fixture
def timing_setup(bench, monkeypatch):
    events = []
    torch_api = SyntheticTorch()
    host = {"hostname": "cpu01", "boot_id": "boot01", "cpu_model": "synthetic CPU", "affinity": [0, 1, 2, 3]}
    tree = ast.parse((ROOT / "scripts/bench_anchor_pilots.py").read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measure_track")
    namespace = {"torch": torch_api, "time": SimpleNamespace(perf_counter=lambda: 0),
                 "np": SimpleNamespace(isfinite=math.isfinite), "stats": linear_stats, "require": bench.require}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "pinned_measure_track", "exec"), namespace)
    helpers = SimpleNamespace(measure_track=namespace["measure_track"], host_info=lambda: copy.deepcopy(host),
                              cv2=SimpleNamespace(getNumThreads=lambda: 1), stats=linear_stats)
    pilots = SimpleNamespace(tensor_sha256=lambda tensors: digest({name: [tensor.value, tensor.dtype, tensor.shape]
                                                                 for name, tensor in tensors.items()}))
    ids = [f"id{index:04d}" for index in range(200)]

    class Dataset:
        def __getitem__(self, index):
            events.append("prepare")
            return {"query": SyntheticTensor((3, 299, 299)), "label": 0, "meta": {"image_id": ids[index]}}

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return SimpleNamespace(core=bench, torch=torch_api, model=SyntheticModel(torch_api, events), dataset=Dataset(),
                           pilots=pilots, bench=helpers, ids=ids, events=events, host=host)


def run_synthetic_timing(worker, setup):
    ticks = iter(index / 1000 for index in range(600))

    def clock():
        setup.events.append("clock")
        return next(ticks)

    return worker.measure_model(setup.model, setup.dataset, setup.ids, core=setup.core, pilots=setup.pilots,
                                bench=setup.bench, torch=setup.torch, expected_shape=[1, 3, 299, 299],
                                expected_host=setup.host, clock=clock)


def test_pinned_timer_materializes_220_scores_inside_network_boundary(worker, timing_setup):
    result = run_synthetic_timing(worker, timing_setup)
    assert timing_setup.events.count("prepare") == 220
    assert timing_setup.events.count("materialize") == 220
    assert timing_setup.events.count("clock") == 600
    assert timing_setup.events[:60] == ["prepare", "network", "materialize"] * 20
    assert timing_setup.events[60:66] == ["clock", "prepare", "clock", "network", "materialize", "clock"]
    assert result["warmups"] == 20 and result["n_documents"] == 200
    assert [row["image_id"] for row in result["per_image"]] == timing_setup.ids
    assert all(row["preparation_ms"] == pytest.approx(1) and row["network_ms"] == pytest.approx(1)
               and row["prepared_input_total_ms"] == pytest.approx(2) for row in result["per_image"])
    assert result["model_before"] == result["model_after"]
    assert result["runtime_before"] == result["runtime_after"]


def test_timing_supports_pinned_torch_21_no_argument_autocast_api(worker, timing_setup):
    timing_setup.torch.is_autocast_enabled = lambda: False
    result = run_synthetic_timing(worker, timing_setup)
    assert result["warmups"] == 20 and result["n_documents"] == 200


@pytest.mark.parametrize("mutation", ["host", "affinity", "environment", "threads", "buffer_precision"])
def test_timing_rechecks_execution_after_all_materializations(worker, timing_setup, monkeypatch, mutation):
    host = copy.deepcopy(timing_setup.host)
    original = timing_setup.dataset

    class Dataset:
        def __getitem__(self, index):
            sample = original[index]
            if index == 199:
                if mutation == "host":
                    timing_setup.bench.host_info = lambda: {**host, "boot_id": "other"}
                elif mutation == "affinity":
                    timing_setup.bench.host_info = lambda: {**host, "affinity": [8, 9, 10, 11]}
                elif mutation == "environment":
                    monkeypatch.setenv("OMP_NUM_THREADS", "2")
                elif mutation == "threads":
                    timing_setup.torch.get_num_interop_threads = lambda: 2
                else:
                    timing_setup.model.buffer.dtype = "float64"
            return sample

    timing_setup.dataset = Dataset()
    with pytest.raises(ValueError):
        run_synthetic_timing(worker, timing_setup)
    assert timing_setup.events.count("materialize") == 220


@pytest.mark.parametrize("mutation", ["parameters", "buffers", "mode", "precision", "host", "threads", "tf32", "amp", "shape", "mask"])
def test_timing_fails_closed_on_model_input_or_execution_drift(worker, timing_setup, mutation):
    if mutation in ("parameters", "buffers", "mode"):
        timing_setup.model.mutation = mutation
    elif mutation == "precision":
        timing_setup.model.parameters_list[0].dtype = "float64"
    elif mutation == "host":
        timing_setup.bench.host_info = lambda: {**timing_setup.host, "boot_id": "other"}
    elif mutation == "threads":
        timing_setup.torch.get_num_threads = lambda: 1
    elif mutation == "tf32":
        timing_setup.torch.backends.cuda.matmul.allow_tf32 = True
    elif mutation == "amp":
        timing_setup.torch.autocast_enabled = True
    else:
        original = timing_setup.dataset

        class BadDataset:
            def __getitem__(self, index):
                sample = original[index]
                if mutation == "shape":
                    sample["query"] = SyntheticTensor((4, 512, 800))
                else:
                    sample["mask"] = SyntheticTensor((1,))
                return sample
        timing_setup.dataset = BadDataset()
    with pytest.raises(ValueError):
        run_synthetic_timing(worker, timing_setup)


@pytest.fixture
def measurement_runner(bench, worker, runtime_catalogue, timing_setup, monkeypatch):
    synthetic_inputs(runtime_catalogue)
    monkeypatch.setattr(bench, "validate_allocation", lambda: None)
    operations = []

    def runner(operation, request_binding, output):
        request = bench.authenticated_json(request_binding, "synthetic request")
        operations.append((operation, request["track"]))
        if operation != "measure":
            process = execute_worker(runtime_catalogue, request_binding, output)
            assert process.returncode == 0, process.stderr
            return
        manifest = bench.authenticated_json(request["runtime_manifest"], "synthetic frozen manifest")
        prepared = copy.deepcopy(manifest["tracks"][request["track"]])
        timing_setup.host.clear()
        timing_setup.host.update(manifest["host"])
        timing_setup.ids[:] = manifest["image_ids"]
        shape = prepared["input_shape"]

        class Dataset:
            def __getitem__(self, index):
                return {"query": SyntheticTensor(shape[1:]), "meta": {"image_id": timing_setup.ids[index]}}

        timing_setup.dataset = Dataset()
        ticks = iter(index / 1000 for index in range(600))
        timing = worker.measure_model(timing_setup.model, Dataset(), timing_setup.ids, core=bench, pilots=timing_setup.pilots,
                                      bench=timing_setup.bench, torch=timing_setup.torch, expected_shape=shape,
                                      expected_host=manifest["host"], clock=lambda: next(ticks))
        prepared.update(timing, operation="measure", status="measured", request_sha256=request_binding["sha256"],
                        runtime_manifest=request["runtime_manifest"], manifest_sha256=manifest["manifest_sha256"],
                        checkpoint_state_sha256=timing["model_before"]["state_sha256"])
        write_json(output, prepared)

    return SimpleNamespace(run=runner, operations=operations, catalogue=runtime_catalogue)


def measurement_binding(tmp_path, catalogue):
    return write_json(tmp_path / "measurement-catalogue.json", catalogue)


def test_measurement_serial_order_frozen_manifest_and_honest_report(bench, measurement_runner, tmp_path):
    binding = measurement_binding(tmp_path, measurement_runner.catalogue)
    output = tmp_path / "timed"
    result = bench.measure_candidate_runtime(binding, output, worker_runner=measurement_runner.run)
    assert result["status"] == "measured" and result["comparable"] is True
    assert measurement_runner.operations == [(operation, track) for operation in ("inspect", "prepare", "measure") for track in TRACKS]
    manifest_binding = authority(output / "runtime-manifest.json")
    bench.validate_candidate_measurement(result, {"catalogue": binding, "runtime_manifest": manifest_binding})
    assert result["winner"] is None and result["family_selection"] == "not-authorized" and result["detector_clearance"] is False
    assert result["full_source_to_decision_latency"] == "not_measured"
    assert result["execution_authorized"] is False
    assert (output / "measurement-source-inventory.json").is_file()
    assert (output / "scientific-source-inventories.json").is_file()
    for track in TRACKS:
        costs = result["excluded_or_unmeasured_costs"][track]
        assert costs["officer_source_to_canonical_registration"] == "not_measured"
        assert costs["offline_legacy_canvas_preparation"] == ("not_applicable_to_query_only_candidate" if track == "R" else "not_measured")
        assert all(value in ("not_measured", "not_applicable_to_query_only_candidate") for value in costs.values())
    markdown = (output / "runtime.md").read_text()
    assert "processed_input_only" in markdown and "not_measured" in markdown
    assert "not a backbone-only" in markdown and "not full source-to-decision" in markdown
    assert json.loads((output / "runtime.json").read_text()) == result
    assert authority(output / "runtime-manifest.json") == manifest_binding


def test_measurement_validator_rejects_binding_statistics_and_execution_drift(bench, measurement_runner, tmp_path):
    binding = measurement_binding(tmp_path, measurement_runner.catalogue)
    output = tmp_path / "timed"
    result = bench.measure_candidate_runtime(binding, output, worker_runner=measurement_runner.run)
    bindings = {"catalogue": binding, "runtime_manifest": authority(output / "runtime-manifest.json")}
    assert result["status"] == "measured"
    mutations = [
        lambda payload: payload["tracks"].pop("N"),
        lambda payload: payload.update(track_order=["N", "R", "D_running_pilot"]),
        lambda payload: payload["tracks"]["R"].update(warmups=19),
        lambda payload: payload["tracks"]["R"].update(n_documents=199),
        lambda payload: payload["tracks"]["R"]["per_image"][0].update(image_id="test-id"),
        lambda payload: payload["tracks"]["R"]["per_image"][0].update(network_ms=float("nan")),
        lambda payload: payload["tracks"]["R"]["per_image"][0].update(network_ms=-1),
        lambda payload: payload["tracks"]["R"]["per_image"][0].update(network_ms=True),
        lambda payload: payload["tracks"]["R"]["per_image"][0].update(prepared_input_total_ms=99),
        lambda payload: payload["tracks"]["R"]["timing"]["network_ms"].update(p95=99),
        lambda payload: payload["tracks"]["R"]["host"].update(boot_id="other"),
        lambda payload: payload["tracks"]["R"]["runtime_after"]["thread_contract"].update(torch_intraop=1),
        lambda payload: payload["tracks"]["R"]["runtime_before"].update(eval_mode=False),
        lambda payload: payload["tracks"]["R"]["runtime_before"]["precision_contract"].update(amp=True),
        lambda payload: payload["tracks"]["R"]["model_after"].update(buffers_sha256="0" * 64),
        lambda payload: payload["tracks"]["R"].update(checkpoint_state_sha256="0" * 64),
        lambda payload: payload["tracks"]["R"].update(scientific_source=payload["tracks"]["D_running_pilot"]["scientific_source"]),
        lambda payload: payload["tracks"]["R"]["sources"][0]["artifacts"][0].update(path="/outside.png"),
        lambda payload: payload["tracks"]["R"].update(input_shape=[1, 4, 512, 800]),
        lambda payload: payload["tracks"]["R"].update(n_total_parameters=999),
        lambda payload: payload.update(winner="R"),
        lambda payload: payload.update(full_source_to_decision_latency=0),
    ]
    for mutate in mutations:
        payload = copy.deepcopy(result)
        mutate(payload)
        with pytest.raises(ValueError):
            bench.validate_candidate_measurement(payload, bindings)
    manifest = json.loads((output / "runtime-manifest.json").read_text())
    manifest["contract"]["warmups"] = 1
    write_json(output / "runtime-manifest.json", manifest)
    with pytest.raises(ValueError, match="SHA256"):
        bench.validate_candidate_measurement(result, bindings)


def test_later_measurement_failure_retains_only_diagnostic_partial_durations(bench, measurement_runner, tmp_path):
    binding = measurement_binding(tmp_path, measurement_runner.catalogue)

    def runner(operation, request, output):
        if operation == "measure" and bench.authenticated_json(request, "request")["track"] == "D_running_pilot":
            raise ValueError("synthetic later worker failure")
        measurement_runner.run(operation, request, output)

    output = tmp_path / "partial"
    result = bench.measure_candidate_runtime(binding, output, worker_runner=runner)
    assert result["status"] == "not-comparable" and result["comparable"] is False and result["measured"] is False
    assert set(result["diagnostic_only"]) == {"R"} and result["tracks"] == {}
    assert "synthetic later worker failure" in result["errors"]
    assert "| R |" not in (output / "runtime.md").read_text()
    assert ("measure", "N") not in measurement_runner.operations


@pytest.mark.parametrize("mutation", ["drift", "missing"])
@pytest.mark.parametrize("failure_track", ["R", "D_running_pilot"])
def test_parent_authority_loss_preserves_failure_reports(bench, measurement_runner, tmp_path, mutation, failure_track):
    binding = measurement_binding(tmp_path, measurement_runner.catalogue)
    parent = Path(measurement_runner.catalogue["candidates"][0]["parent_manifest"]["path"])
    original_sources = json.loads(parent.read_text())["source_files"]

    def runner(operation, request, output):
        measurement_runner.run(operation, request, output)
        if operation != "measure" or bench.authenticated_json(request, "request")["track"] != failure_track:
            return
        if mutation == "drift":
            parent.write_text(parent.read_text() + "\n")
        else:
            parent.unlink()
        if failure_track == "D_running_pilot":
            raise ValueError("synthetic later worker failure with parent loss")

    output = tmp_path / "parent-loss"
    result = bench.measure_candidate_runtime(binding, output, worker_runner=runner)
    assert result["status"] == "not-comparable"
    assert result["comparable"] is False and result["measured"] is False
    assert result["tracks"] == {}
    assert set(result["diagnostic_only"]) == ({"R"} if failure_track == "D_running_pilot" else set())
    assert result["errors"]
    if failure_track == "D_running_pilot":
        assert "synthetic later worker failure with parent loss" in result["errors"]
    elif mutation == "drift":
        assert any("SHA256 mismatch" in error for error in result["errors"])
    assert json.loads((output / "runtime.json").read_text()) == result
    markdown = (output / "runtime.md").read_text()
    assert "Status: not-comparable" in markdown
    assert "| Candidate |" not in markdown and "| R |" not in markdown
    inventories = json.loads((output / "scientific-source-inventories.json").read_text())
    assert inventories["R"]["source_files"] == original_sources
    assert ("measure", "N") not in measurement_runner.operations


def test_measurement_pending_reviews_never_creates_worker_or_output(bench, catalogue, tmp_path):
    binding = measurement_binding(tmp_path, catalogue)
    result = bench.measure_candidate_runtime(binding, tmp_path / "blocked", worker_runner=lambda *args: pytest.fail("worker"))
    assert result["status"] == "waiting-for-completion" and result["comparable"] is False
    assert not (tmp_path / "blocked").exists()


def test_later_worker_cannot_rewrite_an_earlier_measurement_request(bench, measurement_runner, tmp_path):
    binding = measurement_binding(tmp_path, measurement_runner.catalogue)

    def runner(operation, request, output):
        measurement_runner.run(operation, request, output)
        if operation == "measure" and bench.authenticated_json(request, "request")["track"] == "N":
            earlier = output.parent / "measure-R-request.json"
            payload = json.loads(earlier.read_text())
            payload["track"] = "N"
            write_json(earlier, payload)

    result = bench.measure_candidate_runtime(binding, tmp_path / "request-drift", worker_runner=runner)
    assert result["status"] == "not-comparable" and result["comparable"] is False
    assert result["tracks"] == {} and result["diagnostic_only"]


@pytest.fixture
def timing_catalogue(runtime_catalogue):
    definitions = ast.parse(Path(__file__).read_text())
    names = {"SyntheticTensor", "SyntheticTorch", "SyntheticModel", "linear_stats"}
    synthetic_code = "\n".join(ast.get_source_segment(Path(__file__).read_text(), node) for node in definitions.body
                               if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    synthetic_code = '''
import hashlib
import json
import math
import statistics
import sys
from contextlib import contextmanager
from types import SimpleNamespace
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
''' + synthetic_code + '''
torch_api = SyntheticTorch()
sys.modules['torch'] = torch_api
events = []
def tensor_sha256(tensors):
    return digest({name: [tensor.value, tensor.dtype, tensor.shape] for name, tensor in tensors.items()})
def load_state_dict(self, state, strict):
    assert strict is True and set(state) == {'active', 'frozen', 'buffer'}
    self.parameters_list = [state['active'], state['frozen']]
    self.buffer = state['buffer']
def cpu(self):
    return self
def floating(self):
    return self
def evaluation(self):
    self.training = False
    return self
SyntheticModel.load_state_dict = load_state_dict
SyntheticModel.cpu = cpu
SyntheticModel.float = floating
SyntheticModel.eval = evaluation
def fake_clock():
    fake_clock.tick += 1
    return fake_clock.tick / 1000
fake_clock.tick = -1
'''
    for candidate in runtime_catalogue["candidates"]:
        source = Path(candidate["scientific_source"]["directory"])
        package = source / "src/refer_or_clear"
        (package / "synthetic.py").write_text(synthetic_code)
        pilots_path = package / "pilots.py"
        pilots_path.write_text(pilots_path.read_text() + '''
from .synthetic import SyntheticTensor, SyntheticModel, torch_api, events, tensor_sha256
original_read = read_pilot_checkpoint
def read_pilot_checkpoint(path, config, identities):
    saved = original_read(path, config, identities)
    saved['state_dict'] = {name: SyntheticTensor(shape, value, dtype, grad)
                           for name, (shape, value, dtype, grad) in saved['state_dict'].items()}
    return saved
def build_initialized_model(config, weights, expected_sha256):
    assert file_sha256(weights) == expected_sha256
    model = SyntheticModel(torch_api, events)
    model.initialization_provenance = {'pretrained_tensor_sha256': 'a' * 64}
    return model
''')
        query_path = package / "data/query.py"
        query_path.write_text(query_path.read_text() + '''
from ..synthetic import SyntheticTensor
class AnchorQueryDataset:
    def __init__(self, rows, raw_root, legacy_root, track, train, augment):
        assert train is False and augment == {'enabled': False}
        self.rows = rows.to_dict('records')
        assert all(row['role'] == 'val' for row in self.rows)
        self.shape = (3, 299, 299) if track == 'R' else (4, 512, 800)
    def __getitem__(self, index):
        return {'query': SyntheticTensor(self.shape), 'meta': {'image_id': self.rows[index]['image_id']}}
''')
        inventory = [{"path": str(path.relative_to(source)), "sha256": authority(path)["sha256"], "size": path.stat().st_size}
                     for path in sorted(source.rglob("*.py"))]
        candidate["identities"]["source_sha256"] = digest({item["path"]: item["sha256"] for item in inventory})
        candidate["scientific_source"]["inventory_sha256"] = digest(inventory)
        parent_path = Path(candidate["parent_manifest"]["path"])
        parent = json.loads(parent_path.read_text())
        parent.update(runtime_identities=candidate["identities"], source_files=inventory)
        parent["provenance"]["source_sha256"] = digest(inventory)
        candidate["parent_manifest"] = write_json(parent_path, parent)
        checkpoint_path = Path(candidate["selected_checkpoint"]["path"])
        saved = json.loads(checkpoint_path.read_text())
        saved.update(identities=candidate["identities"], initialization_provenance={"pretrained_tensor_sha256": "a" * 64},
                     state_dict={"active": [[12], 1, "float32", True], "frozen": [[3], 1, "float32", False],
                                 "buffer": [[1], 1, "float32", False]})
        candidate["selected_checkpoint"] = write_json(checkpoint_path, saved)
        train = json.loads(Path(candidate["train_json"]["path"]).read_text())
        train.update(identities=candidate["identities"], selected_checkpoint_sha256=candidate["selected_checkpoint"]["sha256"])
        refresh_runtime_ledger(candidate, train)
        candidate["completion_review"]["bindings"]["parent_manifest_sha256"] = candidate["parent_manifest"]["sha256"]
    tools = Path(runtime_catalogue["measurement_tools"]["directory"])
    original = ast.parse((ROOT / "scripts/bench_anchor_pilots.py").read_text())
    timer_code = "\n".join(ast.get_source_segment((ROOT / "scripts/bench_anchor_pilots.py").read_text(), node)
                            for node in original.body if isinstance(node, ast.FunctionDef)
                            and node.name in {"measure_track", "verify_source_artifacts"})
    benchmark_path = tools / "bench_anchor_pilots.py"
    benchmark_path.write_text(benchmark_path.read_text() + '''
import math
from types import SimpleNamespace
from refer_or_clear.synthetic import torch_api as torch, fake_clock, linear_stats as stats
require = sys.modules['anchor_candidate_coordinator'].require
np = SimpleNamespace(isfinite=math.isfinite)
time = SimpleNamespace(perf_counter=fake_clock)
cv2 = SimpleNamespace(getNumThreads=lambda: 1)
''' + timer_code)
    runtime_catalogue["measurement_tools"]["files"][benchmark_path.name] = authority(benchmark_path)["sha256"]
    synthetic_inputs(runtime_catalogue)
    return runtime_catalogue


def test_fresh_workers_measure_only_synthetic_own_source_and_frozen_train_inputs(bench, timing_catalogue, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "validate_allocation", lambda: None)
    binding = measurement_binding(tmp_path, timing_catalogue)
    calls = []

    def runner(operation, request, output):
        calls.append(operation)
        process = execute_worker(timing_catalogue, request, output)
        assert process.returncode == 0, process.stderr

    result = bench.measure_candidate_runtime(binding, tmp_path / "synthetic-timing", worker_runner=runner)
    assert result["status"] == "measured", result["errors"]
    assert calls == ["inspect"] * 3 + ["prepare"] * 3 + ["measure"] * 3
    for track in TRACKS:
        receipt = result["tracks"][track]
        assert receipt["model_before"] == receipt["model_after"]
        assert receipt["checkpoint_state_sha256"] == receipt["model_before"]["state_sha256"]
        assert receipt["n_active_parameters"] == 12 and receipt["n_total_parameters"] == 15
        assert all(row["network_ms"] == pytest.approx(1) for row in receipt["per_image"])
        source = Path(receipt["scientific_source"]["directory"])
        assert all(Path(origin).is_relative_to(source) for origin in receipt["scientific_module_origins"].values())
