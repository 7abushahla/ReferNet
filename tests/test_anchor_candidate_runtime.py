import ast
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bench_anchor_candidates.py"
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


def test_measure_phase_is_disabled_even_when_metadata_ready(bench, catalogue, tmp_path):
    for candidate in catalogue["candidates"]:
        complete(candidate)
    result, output = cli(catalogue, tmp_path, phase="measure")
    assert result.returncode == 1
    assert "not implemented" in result.stderr
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
