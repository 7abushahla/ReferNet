"""Candidate runtime preflight and allocated-only isolated CPU measurement.

The externally SHA256-bound catalogue uses anchor_candidate_bindings_v1.
Candidates retain original parent-manifest identities and scientific sources.
Completion reviews may be text records, with explicit final-evidence bindings
authenticated by the catalogue. Readiness does not verify checkpoint, source,
weight or image bytes and is not a substitute for later scientific inspection.
Measurement setup is not launch approval or family-selection authorization.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess


TRACK_ORDER = ("R", "D_running_pilot", "N")
SCIENTIFIC_REVISIONS = {
    "R": "0959335bd9a6a22510f6706c49c62d84560cd524",
    "N": "0959335bd9a6a22510f6706c49c62d84560cd524",
    "D_running_pilot": "bef7544a64f36a166f76eeef0d3130e353bd3203",
}
SCIENTIFIC_ARCHIVES = {
    "R": "4413528c1afd15f6c420763287bc57d9eb523df2a69138d5b472f0b4b2092ba4",
    "N": "4413528c1afd15f6c420763287bc57d9eb523df2a69138d5b472f0b4b2092ba4",
    "D_running_pilot": "00d0b99e635536d27a01c6f4f39bd94e14e98409dc5616c722f7dd4db3f8d5de",
}
ARCHITECTURES = {
    "R": "anchor_rgb_r50_v1", "D_running_pilot": "anchor_direct_running_ia18_pilot_v1",
    "N": "anchor_absent_ia18_v1",
}
OFFLINE_WEIGHTS = {
    "R": "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    "D_running_pilot": "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618",
    "N": "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618",
}
IDENTITY_FIELDS = {"source_sha256", "config_sha256", "records_sha256", "split_sha256", "weights_sha256"}
EVIDENCE_FILES = {"val_summary.json", "val_predictions.csv", "log.csv", "gate.json", "replay.json"}
ROW_FIELDS = ("image_id", "label", "design_id", "component_id", "path", "attack_family")
ROLE_COUNTS = {"train": 1778, "val": 222, "test": 222}
CONTRACT = {"name": "anchor_candidate_cpu_v1", "track_order": list(TRACK_ORDER), "role": "train", "fold": 0,
            "n_documents": 200, "id_selection": "first_200_lexicographic_train_ids", "warmups": 20,
            "timed_passes": 1, "batch_size": 1, "precision": "float32", "device": "cpu",
            "execution": "eager_inference_mode", "torch_intraop": 4, "torch_interop": 1,
            "opencv_threads": 1, "OMP_NUM_THREADS": 4, "MKL_NUM_THREADS": 4,
            "concurrent_benchmark_workload": False,
            "network_boundary": "ready_tensor_forward_through_attack_score_materialization",
            "preparation_boundary": "source_read_decode_declared_transform_no_upstream_geometry",
            "interpretation": "compound_recipe_preliminary_not_officer_device_end_to_end"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_sha256(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{name} requires a lowercase SHA256")
    return value


def reject_symlinks(path):
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"symlink path forbidden: {component}")


def absolute_path(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} requires an absolute path")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"{name} requires a safe absolute path")
    reject_symlinks(path)
    return path.resolve()


def relative_path(value, name):
    require(isinstance(value, str) and bool(value), f"{name} requires a relative path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and path != Path("."),
            f"{name} path escape forbidden")
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def authenticated_bytes(binding, name):
    require(isinstance(binding, dict), f"{name} requires a path/SHA256 authority")
    path = absolute_path(binding.get("path"), name)
    require(path.suffix in (".json", ".csv", ".py", ".sh", ".md", ".txt"),
            f"{name} must be metadata/tooling, never checkpoint, weight or image bytes")
    expected = validate_sha256(binding.get("sha256"), name)
    require(path.is_file(), f"{name} file missing: {path}")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected, f"{name} SHA256 mismatch: {path}")
    return content


def authenticated_json(binding, name):
    content = authenticated_bytes(binding, name)
    payload = json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)
    require(isinstance(payload, dict), f"{name} JSON object required")
    json_sha256(payload)
    return payload


def validate_tools(tools):
    require(isinstance(tools, dict), "measurement tooling binding required")
    directory = absolute_path(tools.get("directory"), "measurement tools")
    files = tools.get("files")
    require(isinstance(files, dict) and {"bench_anchor_pilots.py", "report_anchor_pilots.py"} <= set(files),
            "measurement bench and inspector authorities required")
    for filename, expected in files.items():
        relative = relative_path(filename, "measurement tool")
        authenticated_bytes({"path": str(directory / relative), "sha256": expected}, "measurement tool")


def read_csv_metadata(binding, name, columns):
    reader = csv.DictReader(io.StringIO(authenticated_bytes(binding, name).decode("utf-8-sig")))
    require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
            f"{name} metadata header missing or duplicated")
    require(set(columns) <= set(reader.fieldnames), f"{name} metadata missing required columns")
    rows = list(reader)
    require(bool(rows), f"{name} metadata empty")
    seen = set()
    for row in rows:
        require(None not in row and all(row.get(column) not in (None, "") for column in columns
                                       if column != "attack_family"), f"{name} metadata malformed/empty fields")
        require(row["image_id"] not in seen, f"{name} metadata duplicate image_id")
        seen.add(row["image_id"])
        require(row["label"] in ("0", "1"), f"{name} metadata binary integer label required")
    return rows


def training_metadata(candidate):
    records = read_csv_metadata(candidate["records"], "records", ROW_FIELDS)
    if "dataset" in records[0]:
        records = [row for row in records if row["dataset"] == "sidtd"]
    split = read_csv_metadata(candidate["split"], "split",
                              ("image_id", "label", "design_id", "component_id", "role", "protocol"))
    require("ref_group" in split[0] or "fold" in split[0], "split metadata official fold binding required")
    require(all(row["protocol"] == "A" for row in split), "split metadata protocol must be A")
    if "ref_group" in split[0]:
        require(all(row["ref_group"] == "it0" for row in split), "split metadata ref_group must be it0")
    if "fold" in split[0]:
        require(all(row["fold"] == "0" for row in split), "split metadata fold must be 0")
    counts = {role: sum(row["role"] == role for row in split) for role in ROLE_COUNTS}
    require(counts == ROLE_COUNTS and len(split) == sum(ROLE_COUNTS.values()),
            "split metadata full official role counts required")
    record_map = {row["image_id"]: row for row in records}
    require(set(record_map) == {row["image_id"] for row in split}, "records/split metadata complete membership mismatch")
    membership = {"train": [], "val": []}
    for row in split:
        record = record_map[row["image_id"]]
        require(all(row[column] == record[column] for column in ("label", "design_id", "component_id")),
                f"records/split metadata join mismatch: {row['image_id']}")
        relative_path(record["path"], "records metadata image")
        if row["role"] in membership:
            membership[row["role"]].append({**{column: record[column] for column in ROW_FIELDS},
                                             "label": int(record["label"])})
    return {role: sorted(rows, key=lambda row: row["image_id"]) for role, rows in membership.items()}


def validate_config(config, track):
    require(isinstance(config, dict), "config object required")
    expected = {"schema_version": "anchor_pilot_v1", "track": track,
                "architecture_id": ARCHITECTURES[track], "fold": 0,
                "seed": 42, "epochs": 100,
                "data_contract": "raw_rgb_299_v1" if track == "R" else "legacy_geometry_diagnostic"}
    for field, value in expected.items():
        require(config.get(field) == value and type(config.get(field)) is type(value), f"config {field} mismatch")
    require(config.get("reference") in (None, "none"), "config reference inputs forbidden")


def validate_parent(candidate):
    track = candidate["track"]
    parent = authenticated_json(candidate.get("parent_manifest"), "parent manifest")
    require(parent.get("schema_version") == "anchor_job_manifest_v1" and parent.get("phase") == "train"
            and parent.get("track") == track, "parent manifest must bind actual train track")
    validate_config(parent.get("effective_config"), track)
    identities = candidate.get("identities")
    require(isinstance(identities, dict) and set(identities) == IDENTITY_FIELDS, "original identities required")
    for field, value in identities.items():
        validate_sha256(value, f"identity {field}")
    require(identities == parent.get("runtime_identities"), "original identities differ from parent manifest")
    provenance = parent.get("provenance")
    require(isinstance(provenance, dict) and all(provenance.get(field) == identities[field]
                                               for field in ("records_sha256", "split_sha256", "weights_sha256")),
            "parent provenance and original runtime identities mismatch")
    require(identities["config_sha256"] == json_sha256(parent["effective_config"]), "config identity mismatch")
    require(identities["weights_sha256"] == OFFLINE_WEIGHTS[track], "offline weight identity mismatch")
    source = candidate.get("scientific_source")
    require(isinstance(source, dict), "scientific source binding required")
    require(source.get("revision") == SCIENTIFIC_REVISIONS[track], "scientific source revision mismatch")
    require(source.get("archive_sha256") == SCIENTIFIC_ARCHIVES[track], "scientific source archive mismatch")
    files = parent.get("source_files")
    require(isinstance(files, list) and bool(files), "scientific source inventory required")
    runtime_files = {}
    seen = set()
    for item in files:
        require(isinstance(item, dict), "scientific source inventory entry invalid")
        relative = relative_path(item.get("path"), "scientific source inventory")
        require(str(relative) not in seen, "scientific source inventory duplicate path")
        seen.add(str(relative))
        validate_sha256(item.get("sha256"), "scientific source inventory")
        if (relative.parts[0] == "src" and relative.suffix == ".py") or str(relative) == "scripts/check_anchor_pilots.py":
            runtime_files[str(relative)] = item["sha256"]
    require("scripts/check_anchor_pilots.py" in runtime_files and any(name.startswith("src/") for name in runtime_files),
            "scientific source runtime inventory incomplete")
    require(json_sha256(runtime_files) == identities["source_sha256"], "scientific source runtime identity mismatch")
    require(source.get("inventory_sha256") == json_sha256(files)
            == parent.get("provenance", {}).get("source_sha256"), "scientific source inventory authority mismatch")
    paths = parent.get("paths")
    require(isinstance(paths, dict), "parent paths required")
    for value in paths.values():
        if value is not None:
            absolute_path(value, "parent path")
    require(absolute_path(source.get("directory"), "scientific source") == absolute_path(paths.get("source_dir"), "parent source"),
            "scientific source directory differs from bound parent")
    run = absolute_path(candidate.get("run_dir"), "candidate run")
    require(run == absolute_path(paths.get("pilot_dir"), "parent pilot")
            and run.parent == absolute_path(paths.get("run_dir"), "parent run")
            and absolute_path(candidate["parent_manifest"]["path"], "parent manifest").parent == run.parent,
            "candidate run/parent manifest linkage mismatch")
    for name in ("raw_root", "legacy_root"):
        require(absolute_path(candidate.get(name), name) == absolute_path(paths.get(name), f"parent {name}"),
                f"{name} differs from parent authority")
    weights = candidate.get("weights")
    require(isinstance(weights, dict) and weights.get("sha256") == identities["weights_sha256"], "weight identity mismatch")
    require(absolute_path(weights.get("path"), "weights") == absolute_path(paths.get("weights"), "parent weights"),
            "weights path differs from bound parent")
    relocations = candidate.get("relocations", {})
    require(isinstance(relocations, dict) and set(relocations) <= {"records", "split"},
            "only explicit metadata relocations supported in preparation")
    for name in ("records", "split"):
        binding = candidate.get(name)
        require(isinstance(binding, dict) and binding.get("sha256") == identities[f"{name}_sha256"],
                f"{name} identity mismatch")
        original = absolute_path(paths.get(name), f"parent {name}")
        requested = absolute_path(binding.get("path"), name)
        relocation = relocations.get(name)
        if requested != original or relocation is not None:
            require(isinstance(relocation, dict)
                    and absolute_path(relocation.get("from"), "relocation original") == original
                    and absolute_path(relocation.get("to"), "relocation destination") == requested
                    and relocation.get("sha256") == identities[f"{name}_sha256"], f"{name} relocation authority mismatch")
    return parent


def validate_membership(rows, expected, name):
    require(isinstance(rows, list) and len(rows) == len(expected), f"{name} membership count mismatch")
    require(all(isinstance(row, dict) and set(ROW_FIELDS) <= set(row) for row in rows), f"{name} membership fields missing")
    canonical = [{field: row[field] for field in ROW_FIELDS} for row in rows]
    require(all(type(row["label"]) is int and row["label"] in (0, 1) for row in canonical),
            f"{name} membership integer labels required")
    require(sorted(canonical, key=lambda row: row["image_id"]) == expected, f"{name} membership differs from metadata")


def validate_completion(candidate, parent, membership):
    train_binding = candidate.get("train_json")
    checkpoint = candidate.get("selected_checkpoint")
    evidence = candidate.get("selected_evidence")
    review = candidate.get("completion_review")
    run = absolute_path(candidate["run_dir"], "candidate run")
    train = None
    if train_binding is not None:
        require(absolute_path(train_binding.get("path"), "train ledger") == run / "train.json", "ledger path linkage mismatch")
        train = authenticated_json(train_binding, "train ledger")
        require(train.get("schema_version") == "anchor_pilot_v1" and train.get("phase") == "train"
                and train.get("track") == candidate["track"] and train.get("architecture_id") == candidate["architecture_id"],
                "ledger track/schema/phase mismatch")
        require(train.get("config") == parent["effective_config"] and train.get("identities") == candidate["identities"],
                "ledger config/original identities mismatch")
        require(type(train.get("epochs_completed")) is int and train["epochs_completed"] == 100
                and type(train.get("selected_epoch")) is int and 1 <= train["selected_epoch"] <= 100,
                "ledger full training/selected epoch required")
        selection = train.get("selection_key")
        require(isinstance(selection, list) and len(selection) == 2
                and all(type(value) in (float, int) and math.isfinite(value) for value in selection),
                "ledger selected key invalid")
        require(train.get("passed") is True and train.get("opened_roles") == ["train", "val"]
                and train.get("no_test_open") is True and train.get("n_images") == {"train": 1778, "val": 222},
                "ledger completion/counts/roles invalid")
        require(isinstance(train.get("paths"), dict), "ledger paths required")
        for value in train["paths"].values():
            if value is not None:
                absolute_path(value, "ledger path")
        for role in ("train", "val"):
            validate_membership(train.get("membership", {}).get(role), membership[role], f"ledger {role}")
    if checkpoint is not None:
        require(isinstance(checkpoint, dict), "selected checkpoint binding required")
        require(absolute_path(checkpoint.get("path"), "selected checkpoint") == run / "checkpoints/best.pt",
                "selected checkpoint must be the bound best.pt, not final/probe")
        validate_sha256(checkpoint.get("sha256"), "selected checkpoint")
        if train is not None:
            require(train.get("selected_checkpoint_sha256") == checkpoint["sha256"], "ledger selected checkpoint mismatch")
    if evidence is not None:
        require(isinstance(evidence, dict) and set(evidence) == EVIDENCE_FILES, "complete selected evidence authorities required")
        for filename, binding in evidence.items():
            require(isinstance(binding, dict) and absolute_path(binding.get("path"), "selected evidence") == run / filename,
                    "selected evidence run linkage mismatch")
            authenticated_bytes(binding, f"selected evidence {filename}")
    if review is not None:
        require(isinstance(review, dict) and review.get("scope") == "final-selected-validation"
                and review.get("approved") is True and isinstance(review.get("reviewer"), str)
                and bool(review["reviewer"].strip()), "review must be independent approved final-selected-validation")
        authenticated_bytes(review, "completion review")
        expected = {"parent_manifest_sha256": candidate["parent_manifest"]["sha256"]}
        if train_binding is not None:
            expected["train_json_sha256"] = train_binding["sha256"]
        if checkpoint is not None:
            expected["selected_checkpoint_sha256"] = checkpoint["sha256"]
        if evidence is not None:
            expected["selected_evidence_sha256"] = {name: binding["sha256"] for name, binding in evidence.items()}
        bindings = review.get("bindings")
        require(isinstance(bindings, dict) and all(bindings.get(name) == value for name, value in expected.items()),
                "review final selected-evidence bindings mismatch")
    return candidate["status"] == "complete" and all(value is not None for value in (train_binding, checkpoint, evidence, review))


def protected_paths(catalogue):
    """Protect original and explicitly relocated inputs, even before readiness."""
    paths = set()

    def protect_file(value):
        paths.add(str(absolute_path(value, "protected input").parent))

    tools = catalogue.get("measurement_tools", {})
    if isinstance(tools, dict) and tools.get("directory") is not None:
        paths.add(str(absolute_path(tools["directory"], "measurement tools")))
    for candidate in catalogue.get("candidates", []):
        require(isinstance(candidate, dict), "candidate object required")
        for name in ("run_dir", "raw_root", "legacy_root"):
            if candidate.get(name) is not None:
                paths.add(str(absolute_path(candidate[name], name)))
        source = candidate.get("scientific_source", {})
        if isinstance(source, dict) and source.get("directory") is not None:
            paths.add(str(absolute_path(source["directory"], "scientific source")))
        for name in ("parent_manifest", "records", "split", "weights", "train_json", "selected_checkpoint", "completion_review"):
            binding = candidate.get(name)
            if isinstance(binding, dict) and binding.get("path") is not None:
                protect_file(binding["path"])
        for binding_name in ("parent_manifest", "train_json"):
            binding = candidate.get(binding_name)
            if isinstance(binding, dict):
                metadata = authenticated_json(binding, f"protected {binding_name}")
                for name, value in metadata.get("paths", {}).items():
                    if value is not None:
                        path = absolute_path(value, "protected original input")
                        paths.add(str(path.parent if name in ("records", "split", "config", "weights") else path))
        for relocation in candidate.get("relocations", {}).values():
            require(isinstance(relocation, dict), "relocation object required")
            for name in ("from", "to"):
                protect_file(relocation.get(name))
    return sorted(paths)


def prepare_bindings(catalogue: dict) -> dict:
    """Validate authenticated metadata; return explicit readiness, never timing.

    Direct callers must authenticate the catalogue externally, as the CLI does.
    Pending entries use explicit null selected authorities. Metadata relocations
    use {kind: {from, to, sha256}} and cannot relocate scientific/data trees.
    """
    result = {"schema_version": "anchor_candidate_readiness_v1", "status": "not-comparable",
              "comparable": False, "execution_authorized": False, "errors": [], "waiting": [],
              "base_timing_contract": "anchor_cpu_v1", "contract": copy.deepcopy(CONTRACT),
              "candidates": {}, "image_ids": [], "full_source_to_decision_latency": "not_measured",
              "excluded_or_unmeasured_costs": ["offline_canvas_rectification_registration", "geometry_target_zone_preparation",
                                                "reference_retrieval_preparation_alignment", "cached_uncached_reference_handling"],
              "verification": {"selected_checkpoint_bytes": False, "scientific_source_bytes": False,
                               "offline_weight_bytes": False, "selected_validation": False, "input_pixels": False}}
    try:
        require(isinstance(catalogue, dict), "candidate catalogue object required")
        json_sha256(catalogue)
        require(catalogue.get("schema_version") == "anchor_candidate_bindings_v1", "candidate catalogue schema mismatch")
        for field, value in (("protocol", "A"), ("fold", 0), ("seed", 42), ("role", "train")):
            require(catalogue.get(field) == value and type(catalogue.get(field)) is type(value), f"catalogue {field} mismatch")
        validate_tools(catalogue.get("measurement_tools"))
        result["measurement_tools"] = copy.deepcopy(catalogue["measurement_tools"])
        entries = catalogue.get("candidates")
        require(isinstance(entries, list) and len(entries) == 3 and all(isinstance(entry, dict) for entry in entries),
                "exactly three candidate objects required")
        require([entry.get("track") for entry in entries] == list(TRACK_ORDER), "candidate track order/duplicates invalid")
        shared_membership = None
        shared_hashes = None
        for candidate in entries:
            track = candidate["track"]
            try:
                require(candidate.get("architecture_id") == ARCHITECTURES[track], "candidate architecture mismatch")
                require(candidate.get("status") in ("pending", "complete", "failed"), "candidate status invalid")
                require(candidate["status"] != "failed", "candidate failed completion")
                for name in ("train_json", "selected_checkpoint", "selected_evidence", "completion_review"):
                    require(name in candidate, f"candidate must explicitly declare {name}, null if pending")
                parent = validate_parent(candidate)
                membership = training_metadata(candidate)
                for role in ("train", "val"):
                    expected = [row["image_id"] for row in membership[role]]
                    observed = parent.get("membership", {}).get(role)
                    require(isinstance(observed, list) and len(observed) == len(expected)
                            and sorted(observed) == expected, f"parent {role} membership mismatch")
                hashes = {name: candidate["identities"][f"{name}_sha256"] for name in ("records", "split")}
                if shared_membership is None:
                    shared_membership, shared_hashes = membership, hashes
                require(membership == shared_membership and hashes == shared_hashes, "common records/split membership identities mismatch")
                image_ids = [row["image_id"] for row in membership["train"][:200]]
                require(catalogue.get("image_ids") == image_ids, "exact first 200 lexicographic training IDs required")
                result["image_ids"] = image_ids
                result["image_ids_sha256"] = json_sha256(image_ids)
                if not validate_completion(candidate, parent, membership):
                    result["waiting"].append(track)
                result["candidates"][track] = copy.deepcopy(candidate)
            except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
                result["errors"].append(f"{track}: {error}")
        result["protected_paths"] = protected_paths(catalogue)
        if not result["errors"]:
            result["status"] = "waiting-for-completion" if result["waiting"] else "metadata-ready"
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        result["errors"].append(str(error))
    return result


def create_output_directory(requested, protections):
    """Reject symlinks before normalization; guard and write the actual path."""
    path = Path(requested).absolute()
    reject_symlinks(path)
    actual = path.resolve()
    for protected in protections:
        root = Path(protected)
        require(not actual.is_relative_to(root) and not root.is_relative_to(actual),
                f"output overlaps protected input tree: {root}")
    require(not actual.exists(), f"output already exists: {actual}")
    actual.mkdir(parents=True, exist_ok=False)
    return actual


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_allocation():
    require(os.environ.get("SLURM_JOB_PARTITION") == "cpu", "allocated CPU partition required")
    require(re.fullmatch(r"[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", "")) is not None,
            "allocated job ID required")
    node = os.environ.get("SLURMD_NODENAME", "")
    require(bool(node) and not any(word in node.lower() for word in ("login", "head", "frontend", "lgn")),
            "allocated non-login CPU node required")
    require(int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) >= 4, "four allocated CPUs required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU visibility must be explicitly disabled")


def authenticate_scientific_source(candidate):
    parent = authenticated_json(candidate["parent_manifest"], "scientific parent manifest")
    root = absolute_path(candidate["scientific_source"]["directory"], "scientific source")
    for path in (root / "src").rglob("*"):
        require(not path.is_symlink(), f"scientific source symlink forbidden: {path}")
        require(path.suffix not in (".pyc", ".pyo", ".so", ".pyd"),
                f"unbound cached/native scientific code forbidden: {path}")
    entries = parent["source_files"]
    expected = {}
    for entry in entries:
        relative = relative_path(entry["path"], "scientific source")
        path = absolute_path(str(root / relative), "scientific source file")
        require(path.is_relative_to(root) and path.is_file(), f"scientific source file missing: {path}")
        require(type(entry.get("size")) is int and entry["size"] == path.stat().st_size,
                f"scientific source size mismatch: {path}")
        require(file_sha256(path) == entry["sha256"], f"scientific source SHA256 mismatch: {path}")
        if (relative.parts[0] == "src" and relative.suffix == ".py") or str(relative) == "scripts/check_anchor_pilots.py":
            expected[str(relative)] = entry["sha256"]
    actual = {}
    for path in sorted((root / "src").rglob("*.py")) + [root / "scripts/check_anchor_pilots.py"]:
        absolute_path(str(path), "scientific runtime file")
        actual[str(path.relative_to(root))] = file_sha256(path)
    require(actual == expected and json_sha256(actual) == candidate["identities"]["source_sha256"],
            "scientific runtime inventory/source SHA256 mismatch")
    return parent


def candidate_fingerprints(catalogue, candidate):
    """Allocated runtime byte verification, never used by metadata preparation."""
    authenticate_scientific_source(candidate)
    paths = {}
    for name in ("parent_manifest", "records", "split", "weights", "train_json", "selected_checkpoint", "completion_review"):
        binding = candidate[name]
        paths[binding["path"]] = binding["sha256"]
    for binding in candidate["selected_evidence"].values():
        paths[binding["path"]] = binding["sha256"]
    tools = catalogue["measurement_tools"]
    for filename, expected in tools["files"].items():
        paths[str(Path(tools["directory"]) / filename)] = expected
    parent = authenticated_json(candidate["parent_manifest"], "scientific parent manifest")
    for entry in parent["source_files"]:
        paths[str(Path(candidate["scientific_source"]["directory"]) / entry["path"])] = entry["sha256"]
    for filename, expected in paths.items():
        path = absolute_path(filename, "immutable runtime authority")
        require(path.is_file() and file_sha256(path) == expected, f"immutable authority SHA256 mismatch: {path}")
    return dict(sorted(paths.items()))


def validate_inspection_result(result, catalogue, candidate, catalogue_sha256):
    require(result.get("schema_version") == "anchor_candidate_worker_result_v1"
            and result.get("operation") == "inspect" and result.get("status") == "inspected",
            "passing candidate inspection required")
    require(result.get("track") == candidate["track"] and result.get("catalogue_sha256") == catalogue_sha256
            and result.get("identities") == candidate["identities"]
            and result.get("scientific_source") == candidate["scientific_source"]
            and result.get("measurement_tools") == catalogue["measurement_tools"], "inspection provenance mismatch")
    require(result.get("comparable") is False and result.get("execution_authorized") is False,
            "inspection cannot authorize timing")
    require(result.get("fingerprints") == candidate_fingerprints(catalogue, candidate), "inspection immutable authorities changed")
    train = authenticated_json(candidate["train_json"], "selected train ledger")
    inspection = result.get("inspection", {})
    require(inspection.get("status") == "complete" and inspection.get("eligible_candidate") is True
            and inspection.get("identities") == candidate["identities"]
            and inspection.get("selected_epoch") == train["selected_epoch"], "selected inspection result mismatch")


def validate_inspection_bundle(bindings, catalogue, catalogue_sha256):
    require(isinstance(bindings, dict) and set(bindings) == set(TRACK_ORDER), "all three inspections required before preparation")
    results = {}
    for candidate in catalogue["candidates"]:
        result = authenticated_json(bindings[candidate["track"]], "candidate inspection")
        validate_inspection_result(result, catalogue, candidate, catalogue_sha256)
        results[candidate["track"]] = result
    require(all(result["host"] == results["R"]["host"] for result in results.values()), "inspection host identity drift")
    return results


def validate_preparation_result(result, catalogue, candidate, catalogue_sha256, inspection):
    require(result.get("schema_version") == "anchor_candidate_worker_result_v1" and result.get("status") == "prepared"
            and result.get("operation") == "prepare" and result.get("track") == candidate["track"]
            and result.get("catalogue_sha256") == catalogue_sha256, "preparation request/provenance mismatch")
    require(result.get("scientific_source") == candidate["scientific_source"]
            and result.get("identities") == candidate["identities"]
            and result.get("measurement_tools") == catalogue["measurement_tools"]
            and result.get("fingerprints") == candidate_fingerprints(catalogue, candidate), "preparation immutable provenance mismatch")
    require(result.get("host") == inspection["host"], "preparation host differs from inspection")
    require(result.get("image_ids") == catalogue["image_ids"]
            and result.get("image_ids_sha256") == json_sha256(catalogue["image_ids"])
            and result.get("membership_role") == "train" and result.get("query_view_role") == "val"
            and result.get("augment") == {"enabled": False}, "preparation training IDs/query view mismatch")
    expected_shape = [1, 3, 299, 299] if candidate["track"] == "R" else [1, 4, 512, 800]
    require(result.get("input_shape") == expected_shape, "prepared input shape mismatch")
    require(result.get("comparable") is False and result.get("execution_authorized") is False,
            "preparation cannot authorize timing")
    train = authenticated_json(candidate["train_json"], "prepared train ledger")
    for field in ("selected_epoch", "selection_key", "n_active_parameters", "n_total_parameters", "initialization"):
        require(result.get(field) == train[field], f"prepared {field} mismatch")
    require(result.get("checkpoint_sha256") == candidate["selected_checkpoint"]["sha256"]
            and result.get("checkpoint_bytes") == inspection["inspection"]["checkpoint_bytes"], "prepared checkpoint mismatch")
    raw = absolute_path(candidate["raw_root"], "raw root")
    legacy = absolute_path(candidate["legacy_root"], "legacy root")
    require(result.get("source_roots") == {"raw": str(raw), "legacy": str(legacy)}, "prepared source roots mismatch")
    require(result.get("preparation_scope") == ("raw_input" if candidate["track"] == "R" else "processed_input_only"),
            "prepared input boundary mismatch")
    for name in ("records", "split", "weights"):
        value = result.get("weights_path") if name == "weights" else result.get("resolved_metadata", {}).get(name)
        path = absolute_path(value, f"prepared {name}")
        require(str(path) in (candidate[name]["path"], train["paths"][name])
                and file_sha256(path) == candidate["identities"][f"{name}_sha256"], f"prepared {name} authority mismatch")
    sources = result.get("sources")
    require(isinstance(sources, list) and len(sources) == 200
            and result.get("sources_sha256") == json_sha256(sources), "prepared source inventory mismatch")
    rows = sorted(train["membership"]["train"], key=lambda row: row["image_id"])[:200]
    for source, row in zip(sources, rows):
        require(source.get("image_id") == row["image_id"], "prepared source/ID mapping mismatch")
        expected = [raw / relative_path(row["path"], "training path")] if candidate["track"] == "R" else [
            legacy / "canvases" / f"{row['image_id']}.png", legacy / "zones" / f"{row['design_id'].replace('/', '_')}.png"]
        artifacts = source.get("artifacts")
        require(isinstance(artifacts, list) and len(artifacts) == len(expected), "prepared source artifact count mismatch")
        for artifact, path in zip(artifacts, expected):
            require(absolute_path(artifact.get("path"), "prepared training artifact") == path
                    and artifact.get("sha256") == file_sha256(path), "prepared training artifact SHA256/path mismatch")


def freeze_candidate_manifest(path, manifest):
    path = Path(path)
    reject_symlinks(path.absolute())
    require(not path.exists(), f"runtime manifest already exists: {path}")
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = json_sha256(payload)
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


def validate_worker_request(receipt, result, expected):
    path = Path(receipt["path"]).parent / f"{expected['operation']}-{expected['track']}-request.json"
    request = authenticated_json({"path": str(path), "sha256": result["request_sha256"]}, "immutable worker request")
    require(request == expected, "worker request provenance mismatch")


def validate_runtime_manifest(binding, catalogue_binding):
    manifest = authenticated_json(binding, "frozen runtime manifest")
    payload = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    require(manifest.get("manifest_sha256") == json_sha256(payload), "canonical runtime manifest SHA256 mismatch")
    require(manifest.get("schema_version") == "anchor_candidate_cpu_v1"
            and manifest.get("base_timing_contract") == "anchor_cpu_v1" and manifest.get("contract") == CONTRACT
            and manifest.get("catalogue") == catalogue_binding, "frozen runtime manifest contract/catalogue mismatch")
    require(manifest.get("comparable") is False and manifest.get("execution_authorized") is False
            and manifest.get("full_source_to_decision_latency") == "not_measured", "manifest cannot authorize comparison")
    catalogue = authenticated_json(catalogue_binding, "original candidate catalogue")
    readiness = prepare_bindings(catalogue)
    require(readiness["status"] == "metadata-ready",
            "all completed selected reviews required: " + "; ".join(readiness["errors"]))
    require(manifest.get("image_ids") == readiness["image_ids"]
            and manifest.get("image_ids_sha256") == readiness["image_ids_sha256"]
            and manifest.get("measurement_tools") == catalogue["measurement_tools"], "manifest original authority drift")
    inspections = validate_inspection_bundle(manifest.get("inspections"), catalogue, catalogue_binding["sha256"])
    require(isinstance(manifest.get("tracks"), dict) and set(manifest["tracks"]) == set(TRACK_ORDER)
            and isinstance(manifest.get("preparations"), dict) and set(manifest["preparations"]) == set(TRACK_ORDER),
            "all prepared candidates required")
    for candidate in catalogue["candidates"]:
        track = candidate["track"]
        validate_worker_request(manifest["inspections"][track], inspections[track], {
            "schema_version": "anchor_candidate_worker_request_v1", "operation": "inspect", "track": track,
            "catalogue": catalogue_binding, "inspections": None})
        prepared = authenticated_json(manifest["preparations"][track], "frozen preparation receipt")
        require(prepared == manifest["tracks"][track], "frozen preparation receipt mismatch")
        validate_preparation_result(prepared, catalogue, candidate, catalogue_binding["sha256"], inspections[track])
        require(prepared["inspections"] == manifest["inspections"] and prepared["host"] == manifest["host"],
                "frozen inspection/host drift")
        validate_worker_request(manifest["preparations"][track], prepared, {
            "schema_version": "anchor_candidate_worker_request_v1", "operation": "prepare", "track": track,
            "catalogue": catalogue_binding, "inspections": manifest["inspections"]})
    return manifest


def timing_statistics(values):
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {"median": median, "p95": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)}


def finite_duration(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative timing required")


def validate_measurement_result(result, manifest, catalogue, candidate, manifest_binding):
    track = candidate["track"]
    prepared = manifest["tracks"][track]
    require(result.get("schema_version") == "anchor_candidate_worker_result_v1" and result.get("operation") == "measure"
            and result.get("status") == "measured" and result.get("track") == track, "measured worker result required")
    for name in ("catalogue_sha256", "identities", "scientific_source", "measurement_tools", "host", "image_ids",
                 "image_ids_sha256", "sources", "sources_sha256", "input_shape", "preparation_scope", "checkpoint_sha256",
                 "checkpoint_bytes", "selected_epoch", "selection_key", "initialization", "n_active_parameters", "n_total_parameters"):
        require(result.get(name) == prepared[name], f"timing frozen {name} mismatch")
    require(result.get("runtime_manifest") == manifest_binding and result.get("manifest_sha256") == manifest["manifest_sha256"],
            "timing manifest authority mismatch")
    require(result.get("comparable") is False and result.get("execution_authorized") is False, "worker cannot authorize comparison")
    require(result.get("fingerprints") == candidate_fingerprints(catalogue, candidate), "timing immutable authority drift")
    expected_runtime = {"host": manifest["host"], "thread_contract": {name: CONTRACT[name] for name in
                        ("torch_intraop", "torch_interop", "opencv_threads", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                        "precision_contract": {"device": "cpu", "precision": "float32", "batch_size": 1, "amp": False, "tf32": False},
                        "inference_mode": True, "eval_mode": True, "CUDA_VISIBLE_DEVICES": ""}
    require(result.get("runtime_before") == expected_runtime and result.get("runtime_after") == expected_runtime
            and result.get("thread_contract") == expected_runtime["thread_contract"]
            and result.get("precision_contract") == expected_runtime["precision_contract"], "timing execution contract drift")
    before = result.get("model_before")
    require(isinstance(before, dict) and set(before) == {"parameters_sha256", "buffers_sha256", "state_sha256"}
            and result.get("model_after") == before, "model parameters/buffers mutated")
    for name, value in before.items():
        validate_sha256(value, name)
    require(result.get("checkpoint_state_sha256") == before["state_sha256"], "selected checkpoint state binding mismatch")
    require(type(result.get("warmups")) is int and result["warmups"] == 20
            and type(result.get("n_documents")) is int and result["n_documents"] == 200, "warmup/document count mismatch")
    rows = result.get("per_image")
    fields = ("preparation_ms", "network_ms", "prepared_input_total_ms")
    require(isinstance(rows, list) and len(rows) == 200, "exactly 200 timed records required")
    for row, image_id in zip(rows, manifest["image_ids"]):
        require(isinstance(row, dict) and set(row) == {"image_id", *fields} and row["image_id"] == image_id,
                "timed record ID/source relationship mismatch")
        for field in fields:
            finite_duration(row[field])
        require(math.isclose(row["preparation_ms"] + row["network_ms"], row["prepared_input_total_ms"], rel_tol=1e-10, abs_tol=1e-9),
                "timing total accounting mismatch")
    timing = result.get("timing")
    require(isinstance(timing, dict) and set(timing) == set(fields), "timing summary fields mismatch")
    for field in fields:
        summary = timing[field]
        require(isinstance(summary, dict) and set(summary) == {"median", "p95"}, "timing percentile fields mismatch")
        for name, expected in timing_statistics([row[field] for row in rows]).items():
            finite_duration(summary[name])
            require(math.isclose(summary[name], expected, rel_tol=1e-10, abs_tol=1e-9), "independently recomputed percentile mismatch")


def excluded_candidate_costs():
    names = ("officer_source_to_canonical_registration", "detector_and_crop", "zone_reference_preparation",
             "enrollment_and_reference_loading", "reference_retrieval", "reference_alignment", "reference_cache_handling",
             "query_reference_matching", "production_fusion", "officer_device_execution", "metadata_loading",
             "model_initialization_and_checkpoint_loading")
    return {track: dict({name: "not_measured" for name in names},
                        offline_legacy_canvas_preparation="not_applicable_to_query_only_candidate" if track == "R" else "not_measured")
            for track in TRACK_ORDER}


def validate_candidate_measurement(measurement: dict, bindings: dict) -> None:
    manifest = validate_runtime_manifest(bindings["runtime_manifest"], bindings["catalogue"])
    catalogue = authenticated_json(bindings["catalogue"], "original candidate catalogue")
    require(measurement.get("schema_version") == "anchor_candidate_measurement_v1"
            and measurement.get("status") == "measured" and measurement.get("measured") is True
            and measurement.get("comparable") is True and measurement.get("track_order") == list(TRACK_ORDER),
            "all three serial measurements required")
    require(measurement.get("catalogue") == bindings["catalogue"]
            and measurement.get("runtime_manifest") == bindings["runtime_manifest"]
            and measurement.get("manifest_sha256") == manifest["manifest_sha256"] and measurement.get("contract") == CONTRACT
            and measurement.get("host") == manifest["host"], "measurement original/frozen binding mismatch")
    require(measurement.get("winner") is None and measurement.get("family_selection") == "not-authorized"
            and measurement.get("detector_clearance") is False and measurement.get("execution_authorized") is False
            and measurement.get("full_source_to_decision_latency") == "not_measured"
            and measurement.get("excluded_or_unmeasured_costs") == excluded_candidate_costs(), "unsupported clearance/latency claim")
    require(measurement.get("errors") == [] and measurement.get("diagnostic_only") == {}, "partial timing cannot be comparable")
    tracks = measurement.get("tracks")
    require(isinstance(tracks, dict) and set(tracks) == set(TRACK_ORDER), "all three measurements required")
    require(isinstance(measurement.get("receipts"), dict) and set(measurement["receipts"]) == set(TRACK_ORDER),
            "all three immutable measurement receipts required")
    for candidate in catalogue["candidates"]:
        track = candidate["track"]
        validate_measurement_result(tracks[track], manifest, catalogue, candidate, bindings["runtime_manifest"])
        receipt = measurement["receipts"][track]
        require(authenticated_json(receipt, "immutable measurement receipt") == tracks[track], "measurement receipt changed")
        validate_worker_request(receipt, tracks[track], {
            "schema_version": "anchor_candidate_worker_request_v1", "operation": "measure", "track": track,
            "catalogue": bindings["catalogue"], "inspections": manifest["inspections"], "runtime_manifest": bindings["runtime_manifest"]})


def run_candidate_worker(operation, request_binding, output):
    request = authenticated_json(request_binding, "worker request")
    catalogue = authenticated_json(request["catalogue"], "worker catalogue")
    tools = catalogue["measurement_tools"]
    validate_tools(tools)
    require("bench_anchor_candidate_worker.py" in tools["files"], "authenticated scientific worker required")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    process = subprocess.run([sys.executable, "-I", str(Path(tools["directory"]) / "bench_anchor_candidate_worker.py"),
                              "--operation", operation, "--request", request_binding["path"],
                              "--request-sha256", request_binding["sha256"], "--out", str(output)],
                             env=environment, capture_output=True, text=True)
    for suffix, content in (("stdout", process.stdout), ("stderr", process.stderr)):
        with Path(str(output) + f".{suffix}.log").open("x") as stream:
            stream.write(content)
    require(process.returncode == 0, f"{operation} worker failed: {process.stderr.strip()}")


def prepare_candidate_runtime(catalogue_binding, run_dir, worker_runner=None):
    """Inspect all candidates serially, then prepare and freeze; never measure.

    worker_runner(operation, request_authority, output_path) writes one result.
    The default uses this interpreter in isolated fresh processes. The stable
    prepare CLI remains metadata-only. Actual timing requires launch approval.
    """
    catalogue = authenticated_json(catalogue_binding, "candidate catalogue")
    readiness = prepare_bindings(catalogue)
    if readiness["status"] != "metadata-ready":
        return readiness
    validate_allocation()
    output = create_output_directory(run_dir, protected_paths(catalogue) + [catalogue_binding["path"]])
    runner = worker_runner or run_candidate_worker
    result = dict(readiness, status="not-comparable", inspections={}, preparations={})
    try:
        for operation in ("inspect", "prepare"):
            for candidate in catalogue["candidates"]:
                track = candidate["track"]
                request = {"schema_version": "anchor_candidate_worker_request_v1", "operation": operation,
                           "track": track, "catalogue": catalogue_binding,
                           "inspections": result["inspections"] if operation == "prepare" else None}
                request_path = output / f"{operation}-{track}-request.json"
                with request_path.open("x") as stream:
                    json.dump(request, stream, sort_keys=True, indent=2, allow_nan=False)
                request_binding = {"path": str(request_path), "sha256": file_sha256(request_path)}
                result_path = output / f"{operation}-{track}-result.json"
                runner(operation, request_binding, result_path)
                binding = {"path": str(result_path), "sha256": file_sha256(result_path)}
                payload = authenticated_json(binding, f"{operation} result")
                require(payload.get("request_sha256") == request_binding["sha256"], "worker request/result binding mismatch")
                if operation == "inspect":
                    validate_inspection_result(payload, catalogue, candidate, catalogue_binding["sha256"])
                    result["inspections"][track] = binding
                else:
                    inspected = authenticated_json(result["inspections"][track], "candidate inspection")
                    validate_preparation_result(payload, catalogue, candidate, catalogue_binding["sha256"], inspected)
                    result["preparations"][track] = binding
            validate_inspection_bundle(result["inspections"], catalogue, catalogue_binding["sha256"])
        tracks = {track: authenticated_json(binding, "prepared candidate") for track, binding in result["preparations"].items()}
        for candidate in catalogue["candidates"]:
            inspected = authenticated_json(result["inspections"][candidate["track"]], "candidate inspection")
            validate_preparation_result(tracks[candidate["track"]], catalogue, candidate, catalogue_binding["sha256"], inspected)
        host = tracks["R"]["host"]
        require(all(item["host"] == host for item in tracks.values()), "prepared host identity drift")
        manifest = {"schema_version": "anchor_candidate_cpu_v1", "base_timing_contract": "anchor_cpu_v1",
                    "contract": copy.deepcopy(CONTRACT), "catalogue": catalogue_binding,
                    "measurement_tools": catalogue["measurement_tools"], "image_ids": readiness["image_ids"],
                    "image_ids_sha256": readiness["image_ids_sha256"], "host": host, "tracks": tracks,
                    "inspections": result["inspections"], "preparations": result["preparations"],
                    "full_source_to_decision_latency": "not_measured", "comparable": False,
                    "execution_authorized": False, "excluded_or_unmeasured_costs": readiness["excluded_or_unmeasured_costs"]}
        frozen = freeze_candidate_manifest(output / "runtime-manifest.json", manifest)
        result.update(status="prepared", manifest_sha256=frozen["manifest_sha256"])
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, AttributeError) as error:
        result["errors"].append(str(error))
    with (output / "runtime-preparation.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    return result


def write_new_json(path, value):
    reject_symlinks(Path(path).absolute())
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_markdown(result):
    lines = ["# Preliminary candidate CPU timing", "", f"Status: {result['status']}", "",
             "These are compound-recipe measurements, not a backbone-only comparison.",
             "Same-host comparability does not establish officer-device deployment performance.",
             "Input-plus-network timing is not full source-to-decision deployment latency.",
             "Full source-to-decision latency: not_measured. No detector clearance or family selection is authorized.", ""]
    if result["comparable"]:
        lines.extend(["| Candidate | Input boundary | Preparation median/p95 ms | Network median/p95 ms | Input + network median/p95 ms |",
                      "| --- | --- | --- | --- | --- |"])
        for track in TRACK_ORDER:
            measured = result["tracks"][track]
            cells = [f"{measured['timing'][name]['median']:.6g}/{measured['timing'][name]['p95']:.6g}" for name in
                     ("preparation_ms", "network_ms", "prepared_input_total_ms")]
            lines.append(f"| {track} | {measured['preparation_scope']} | {' | '.join(cells)} |")
    else:
        lines.append("Partial durations are diagnostic-only and excluded from comparison tables.")
    lines.extend(["", "## Excluded or unmeasured costs", "",
                  "R includes raw decoding and resize. D/N start from precomputed canvases and zones; offline registration is excluded."])
    for track, costs in result["excluded_or_unmeasured_costs"].items():
        lines.append(f"\n{track}")
        lines.extend(f"- {name}: {status}" for name, status in costs.items())
    if result["errors"]:
        lines.extend(["", "## Errors", *[f"- {error}" for error in result["errors"]]])
    return "\n".join(lines) + "\n"


def measure_candidate_runtime(catalogue_binding, run_dir, worker_runner=None):
    """Allocated-only serial execution. Preparation is not launch authorization."""
    preparation = prepare_candidate_runtime(catalogue_binding, run_dir, worker_runner)
    if "inspections" not in preparation:
        return preparation
    output = Path(run_dir).resolve()
    catalogue = authenticated_json(catalogue_binding, "original candidate catalogue")
    result = {"schema_version": "anchor_candidate_measurement_v1", "status": "not-comparable", "measured": False,
              "comparable": False, "execution_authorized": False, "winner": None, "family_selection": "not-authorized",
              "detector_clearance": False, "full_source_to_decision_latency": "not_measured",
              "excluded_or_unmeasured_costs": excluded_candidate_costs(), "catalogue": catalogue_binding,
              "contract": copy.deepcopy(CONTRACT), "track_order": list(TRACK_ORDER), "tracks": {},
              "diagnostic_only": {}, "receipts": {}, "errors": list(preparation["errors"])}
    runner = worker_runner or run_candidate_worker
    scientific_inventories = {}
    try:
        require(preparation["status"] == "prepared", "all inspections/preparations required before timing")
        for candidate in catalogue["candidates"]:
            parent = authenticated_json(candidate["parent_manifest"], "scientific parent")
            scientific_inventories[candidate["track"]] = {
                "scientific_source": candidate["scientific_source"], "source_files": parent["source_files"]}
        manifest_binding = {"path": str(output / "runtime-manifest.json"), "sha256": file_sha256(output / "runtime-manifest.json")}
        manifest = validate_runtime_manifest(manifest_binding, catalogue_binding)
        result.update(runtime_manifest=manifest_binding, manifest_sha256=manifest["manifest_sha256"], host=manifest["host"])
        receipts = {}
        for candidate in catalogue["candidates"]:
            validate_allocation()
            validate_runtime_manifest(manifest_binding, catalogue_binding)
            track = candidate["track"]
            request = {"schema_version": "anchor_candidate_worker_request_v1", "operation": "measure", "track": track,
                       "catalogue": catalogue_binding, "inspections": manifest["inspections"], "runtime_manifest": manifest_binding}
            request_path = output / f"measure-{track}-request.json"
            write_new_json(request_path, request)
            request_binding = {"path": str(request_path), "sha256": file_sha256(request_path)}
            result_path = output / f"measure-{track}-result.json"
            runner("measure", request_binding, result_path)
            authenticated_json(request_binding, "immutable measurement request")
            receipt = {"path": str(result_path), "sha256": file_sha256(result_path)}
            payload = authenticated_json(receipt, "measured candidate")
            require(payload.get("request_sha256") == request_binding["sha256"], "measurement request/result binding mismatch")
            validate_runtime_manifest(manifest_binding, catalogue_binding)
            validate_measurement_result(payload, manifest, catalogue, candidate, manifest_binding)
            receipts[track] = receipt
            result["diagnostic_only"][track] = payload
        for track, receipt in receipts.items():
            require(authenticated_json(receipt, "immutable measured receipt") == result["diagnostic_only"][track],
                    "earlier measurement receipt mutated")
        result.update(status="measured", measured=True, comparable=True, tracks=result["diagnostic_only"], diagnostic_only={}, receipts=receipts)
        validate_candidate_measurement(result, {"catalogue": catalogue_binding, "runtime_manifest": manifest_binding})
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, AttributeError) as error:
        if result["tracks"]:
            result["diagnostic_only"] = result["tracks"]
        result.update(status="not-comparable", measured=False, comparable=False, tracks={})
        result["errors"].append(str(error))
    write_new_json(output / "measurement-source-inventory.json", catalogue["measurement_tools"])
    write_new_json(output / "scientific-source-inventories.json", scientific_inventories)
    write_new_json(output / "runtime.json", result)
    with (output / "runtime.md").open("x") as stream:
        stream.write(runtime_markdown(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "measure"), required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidates-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        binding = {"path": str(Path(args.candidates).absolute()), "sha256": args.candidates_sha256}
        if args.phase == "measure":
            result = measure_candidate_runtime(binding, args.run_dir)
            print(json.dumps({"status": result["status"], "comparable": result["comparable"], "execution_authorized": False,
                              "report": str(Path(args.run_dir) / "runtime.json") if Path(args.run_dir).exists() else None}))
            return 0 if result["status"] == "measured" else 2 if result["status"] == "waiting-for-completion" else 1
        catalogue = authenticated_json(binding, "candidate catalogue")
        result = prepare_bindings(catalogue)
        protections = protected_paths(catalogue)
        protections.append(str(absolute_path(binding["path"], "candidate catalogue")))
        output = create_output_directory(args.run_dir, protections)
        result["catalogue_sha256"] = binding["sha256"]
        with (output / "readiness.json").open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "report": str(output / "readiness.json"),
                          "comparable": False, "execution_authorized": False}))
        return {"metadata-ready": 0, "waiting-for-completion": 2, "not-comparable": 1}[result["status"]]
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
