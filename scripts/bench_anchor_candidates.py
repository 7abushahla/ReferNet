"""Metadata-only candidate runtime preparation. No measurement is authorized here.

The externally SHA256-bound catalogue uses anchor_candidate_bindings_v1.
Candidates retain original parent-manifest identities and scientific sources.
Completion reviews may be text records, with explicit final-evidence bindings
authenticated by the catalogue. Readiness does not verify checkpoint, source,
weight or image bytes and is not a substitute for later scientific inspection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "measure"), required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidates-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        require(args.phase == "prepare", "measurement is not implemented or authorized by Task 1 preparation")
        binding = {"path": str(Path(args.candidates).absolute()), "sha256": args.candidates_sha256}
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
