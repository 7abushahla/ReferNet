"""Freeze bounded pilot inputs and validate coordinator-provided review evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import types
import xml.etree.ElementTree as ET


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve_config(source_dir, config):
    namespace = "_anchor_job_snapshot"
    package = types.ModuleType(namespace)
    package.__path__ = [str(source_dir / "src/refer_or_clear")]
    sys.modules[namespace] = package
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        resolver = importlib.import_module(f"{namespace}.pilot_config")
        return resolver.resolve_pilot_config(config)
    finally:
        sys.dont_write_bytecode = previous
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]


def runtime_source_sha256(entries):
    return object_sha256({entry["path"]: entry["sha256"] for entry in entries
                          if (entry["path"].startswith("src/") and entry["path"].endswith(".py"))
                          or entry["path"] == "scripts/check_anchor_pilots.py"})


def relative_path(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe input path {value!r}")
    if any(part.startswith("._") for part in path.parts):
        raise ValueError(f"sidecar input path {value!r}")
    return path


def input_file(root, relative):
    relative = relative_path(str(relative))
    path = root / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"symlink input path {path}")
    if not path.is_file():
        raise FileNotFoundError(f"required input path {path}")
    return path


def source_entries(root):
    entries = []
    ignored = {".git", "__pycache__", ".pytest_cache"}
    source_folders = {"src", "scripts", "configs", "tests", "docs"}
    source_suffixes = {".py", ".sh", ".yaml", ".yml", ".toml", ".txt", ".md", ".json", ".cfg", ".ini", ".lock"}
    for directory, folders, filenames in os.walk(root, followlinks=False):
        folders[:] = sorted(name for name in folders if name not in ignored and not name.startswith("._"))
        if Path(directory) == root:
            folders[:] = [name for name in folders if name in source_folders]
        for name in folders:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"symlink source path {Path(directory) / name}")
        for name in sorted(filenames):
            if name.startswith("._") or Path(name).suffix.lower() not in source_suffixes:
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid source path {path}")
            entries.append(dict(path=path.relative_to(root).as_posix(), sha256=file_sha256(path),
                                size=path.stat().st_size))
    if not entries:
        raise ValueError("source snapshot is empty")
    return sorted(entries, key=lambda entry: entry["path"])


def read_rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"metadata path {path} lacks {sorted(required)}")
        rows = list(reader)
    indexed = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed metadata path {path}")
        identifier = row["image_id"]
        if not identifier or identifier in indexed:
            raise ValueError(f"empty or duplicate image_id in {path}")
        indexed[identifier] = row
    return indexed


def selected_rows(records, split, track):
    inventory = read_rows(records, {"image_id", "dataset", "path", "design_id", "label"})
    roles = read_rows(split, {"image_id", "role"})
    forbidden_raw = {inventory[identifier]["path"] for identifier, membership in roles.items()
                     if identifier in inventory and membership["role"] not in {"train", "val"}}
    selected = []
    for identifier, membership in sorted(roles.items()):
        if identifier not in inventory:
            raise ValueError(f"split mismatch for image_id {identifier}")
        record = inventory[identifier]
        for field in (record.keys() & membership.keys()) - {"image_id"}:
            if record[field] != membership[field]:
                raise ValueError(f"metadata mismatch for {identifier} field {field}")
        if membership["role"] not in {"train", "val"}:
            continue
        if record["dataset"].lower() != "sidtd" or record["label"] not in {"0", "1"}:
            raise ValueError(f"invalid pilot membership for {identifier}")
        if len(relative_path(identifier).parts) != 1 or "\\" in identifier:
            raise ValueError(f"unsafe image_id path {identifier}")
        if track == "R":
            raw_path = relative_path(record["path"])
            forbidden_folders = {"test", "tests", "clip", "clips", "video", "videos",
                                 "refs", "references", "reference_only"}
            if (raw_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
                    or record["path"] in forbidden_raw
                    or forbidden_folders.intersection(part.lower() for part in raw_path.parts[:-1])):
                raise ValueError(f"forbidden raw test/reference/clip path {raw_path}")
        selected.append(dict(record, role=membership["role"]))
    if {row["role"] for row in selected} != {"train", "val"}:
        raise ValueError("pilot inputs require both train and val membership")
    return selected


def data_entries(rows, track, raw_root, legacy_root):
    needed = set()
    for row in rows:
        identifier = row["image_id"]
        if track == "R":
            needed.add(("raw", relative_path(row["path"]).as_posix()))
        else:
            needed.add(("legacy", f"canvases/{identifier}.png"))
            zone = row["design_id"].replace("/", "_")
            if not zone or "\\" in zone:
                raise ValueError(f"unsafe design zone path {zone}")
            needed.add(("legacy", relative_path(f"zones/{zone}.png").as_posix()))
            if track == "N" and row["role"] == "train":
                needed.add(("legacy", f"valid/{identifier}.png"))
                if row["label"] == "1":
                    needed.add(("legacy", f"masks/{identifier}.png"))
    roots = {"raw": raw_root, "legacy": legacy_root}
    entries = []
    for kind, relative in sorted(needed):
        path = input_file(roots[kind], relative)
        entries.append(dict(kind=kind, path=relative, sha256=file_sha256(path), size=path.stat().st_size))
    return entries


def copy_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(destination) != expected or file_sha256(source) != expected:
        raise ValueError(f"input hash changed during staging {source}")
    destination.chmod(0o444)


def freeze_directories(root):
    for directory, _, _ in os.walk(root, topdown=False):
        Path(directory).chmod(0o555)


def passing_gate(payload, identities):
    if (not isinstance(payload, dict) or payload.get("phase") != "pretrain"
            or payload.get("passed") is not True or payload.get("failures")
            or payload.get("identities") != identities):
        raise ValueError("passing pretrain gate with matching runtime identities required")


def check_stack_report(report):
    cases = ET.parse(report).getroot().findall(".//testcase")
    if (len(cases) != 1 or cases[0].get("name") != "test_actual_opencd_gpu_integration"
            or any(cases[0].find(tag) is not None for tag in ("skipped", "failure", "error"))):
        raise ValueError("actual Open-CD GPU integration must pass without a skip")


def check_audit_health(run_dir):
    audit = run_dir / "audit"
    try:
        summary = json.loads((audit / "summary.json").read_text())
        completeness = summary.get("completeness", {})
        required = ("complete", "candidate_headers_complete", "manifest_membership_complete",
                    "exhaustive_test_inventory_confirmed")
        passed = (summary.get("schema_version") == 2 and "candidate_test" in summary.get("groups", {})
                  and summary.get("disk_inventory", {}).get("scanned") is True
                  and all(completeness.get(key) is True for key in required)
                  and not completeness.get("reasons"))
        reasons = completeness.get("reasons", [])
        if not passed and not reasons:
            reasons = ["Schema-2 exhaustive candidate header/inventory completeness is unverified."]
    except (OSError, ValueError, TypeError, AttributeError) as error:
        passed, reasons = False, [f"Audit summary unavailable or invalid: {error}"]
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "health.json").write_text(json.dumps(dict(passed=passed, reasons=reasons), indent=2) + "\n")
    if not passed:
        raise ValueError("encoding header audit is incomplete; see audit/health.json")


def validated_review(review, provenance, identities):
    if review is None:
        raise ValueError("train requires an explicit independent review record")
    review = Path(review).resolve(strict=True)
    review_hash = file_sha256(review)
    payload = json.loads(review.read_text())
    if (not isinstance(payload, dict) or payload.get("schema_version") != "anchor_review_v1"
            or payload.get("approved") is not True
            or not isinstance(payload.get("reviewer"), str) or not payload["reviewer"].strip()
            or not isinstance(payload.get("reviewed_at"), str) or not payload["reviewed_at"].strip()
            or payload.get("provenance") != provenance):
        raise ValueError("review approval and matching provenance required")
    gate = Path(payload.get("gate_path", ""))
    if not gate.is_absolute() or not gate.is_file():
        raise ValueError("review requires an absolute existing gate_path")
    gate_hash = file_sha256(gate)
    if gate_hash != payload.get("gate_sha256"):
        raise ValueError("review gate hash mismatch")
    passing_gate(json.loads(gate.read_text()), identities)
    manifest_path = gate.parent.parent / "input_manifest.json"
    if gate.name != "gate.json" or gate.parent.name != "pilot" or not manifest_path.is_file():
        raise ValueError("review gate must belong to a packaged diagnostic pilot")
    manifest_hash = file_sha256(manifest_path)
    diagnostic = json.loads(manifest_path.read_text())
    if (diagnostic.get("phase") != "diagnostic" or diagnostic.get("provenance") != provenance
            or diagnostic.get("runtime_identities") != identities):
        raise ValueError("review diagnostic manifest provenance/identities mismatch")
    return review, review_hash, gate, gate_hash, manifest_path, manifest_hash


def environment_snapshot():
    versions = {}
    for package in ("torch", "torchvision", "numpy", "opencv-python-headless", "pandas", "PyYAML",
                    "Pillow", "pytest", "mmcv", "mmengine", "mmsegmentation", "mmdet", "mmpretrain", "opencd"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(conda_prefix=os.environ.get("CONDA_PREFIX"), python_executable=sys.executable,
                python_version=platform.python_version(), platform=platform.platform(), packages=versions,
                flags={key: os.environ.get(key) for key in
                       ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "PYTHONHASHSEED")})


def prepare_job(*, source_dir, config, raw_root, legacy_root, records, split, weights,
                weights_sha256, stage_dir, run_dir, phase, review):
    if phase not in {"diagnostic", "train", "audit"}:
        raise ValueError("unknown job phase")
    source_dir = Path(source_dir).resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("source directory required")
    stage_dir, run_dir = Path(stage_dir).resolve(), Path(run_dir).resolve()
    if stage_dir.exists() or run_dir.exists():
        raise FileExistsError("new scratch and output directories required")
    if (stage_dir.is_relative_to(source_dir) or run_dir.is_relative_to(source_dir)
            or stage_dir.is_relative_to(run_dir) or run_dir.is_relative_to(stage_dir)):
        raise ValueError("source, scratch and output paths must be isolated")
    records, split = Path(records).resolve(strict=True), Path(split).resolve(strict=True)
    raw_root, legacy_root = Path(raw_root).resolve(), Path(legacy_root).resolve()
    metadata = {"records.csv": records, "split.csv": split}
    source_files = source_entries(source_dir)
    provenance = dict(source_sha256=object_sha256(source_files),
                      records_sha256=file_sha256(records), split_sha256=file_sha256(split))
    track, membership, files, identities, effective_config = None, {}, [], None, None
    review_files = None
    if phase != "audit":
        config, weights = Path(config).resolve(strict=True), Path(weights).resolve(strict=True)
        config_bytes = config.read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        settings = resolve_config(source_dir, config)
        if (not isinstance(settings, dict) or settings.get("schema_version") != "anchor_pilot_v1"
                or settings.get("track") not in {"R", "D", "N"}):
            raise ValueError("anchor_pilot_v1 config with track R/D/N required")
        track = settings["track"]
        effective_config = settings
        if file_sha256(weights) != weights_sha256:
            raise ValueError("pretrained weights hash mismatch")
        rows = selected_rows(records, split, track)
        files = data_entries(rows, track, raw_root, legacy_root)
        membership = {role: [row["image_id"] for row in rows if row["role"] == role]
                      for role in ("train", "val")}
        provenance.update(config_sha256=config_hash, weights_sha256=weights_sha256,
                          data_sha256=object_sha256(files))
        identities = dict(source_sha256=runtime_source_sha256(source_files), config_sha256=object_sha256(settings),
                          records_sha256=provenance["records_sha256"], split_sha256=provenance["split_sha256"],
                          weights_sha256=weights_sha256)
        metadata.update({"config.yaml": config, "weights.pth": weights})
        if phase == "train":
            review_files = validated_review(review, provenance, identities)
    else:
        if not raw_root.is_dir():
            raise ValueError("audit raw root directory required")
        provenance["data_sha256"] = object_sha256({key: provenance[key] for key in
                                                  ("records_sha256", "split_sha256")})
    stage_dir.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("raw", "legacy", "code", "inputs"):
        (stage_dir / directory).mkdir()
    for entry in source_files:
        copy_verified(source_dir / entry["path"], stage_dir / "code" / entry["path"], entry["sha256"])
    if object_sha256(source_entries(source_dir)) != provenance["source_sha256"]:
        raise ValueError("source hash changed during staging")
    hashes = {"records.csv": provenance["records_sha256"], "split.csv": provenance["split_sha256"]}
    if phase != "audit":
        hashes.update({"config.yaml": provenance["config_sha256"], "weights.pth": weights_sha256})
    for filename, source in metadata.items():
        copy_verified(source, stage_dir / "inputs" / filename, hashes[filename])
        if filename != "weights.pth":
            copy_verified(source, run_dir / "inputs" / filename, hashes[filename])
    for entry in files:
        root = raw_root if entry["kind"] == "raw" else legacy_root
        copy_verified(input_file(root, entry["path"]), stage_dir / entry["kind"] / entry["path"], entry["sha256"])
    if review_files:
        review_path, review_hash, gate_path, gate_hash, diagnostic_path, diagnostic_hash = review_files
        copy_verified(review_path, run_dir / "inputs/review.json", review_hash)
        copy_verified(gate_path, run_dir / "inputs/pretrain_gate.json", gate_hash)
        copy_verified(gate_path, stage_dir / "inputs/pretrain_gate.json", gate_hash)
        copy_verified(diagnostic_path, run_dir / "inputs/diagnostic_manifest.json", diagnostic_hash)
    for directory in ("raw", "legacy", "code", "inputs"):
        freeze_directories(stage_dir / directory)
    freeze_directories(run_dir / "inputs")
    manifest = dict(schema_version="anchor_job_manifest_v1", phase=phase, track=track,
                    provenance=provenance, runtime_identities=identities, effective_config=effective_config,
                    membership=membership, source_files=source_files, data_files=files,
                    data_scope="metadata_only" if phase == "audit" else "train_val_only",
                    paths=dict(source_dir=str(source_dir), raw_root=str(raw_root), legacy_root=str(legacy_root),
                               records=str(records), split=str(split), config=str(config) if config else None,
                               weights=str(weights) if weights else None, stage_dir=str(stage_dir), run_dir=str(run_dir),
                               pilot_dir=str(run_dir / "pilot")),
                    allocation=dict(job_id=os.environ.get("SLURM_JOB_ID"),
                                    node=os.environ.get("SLURMD_NODENAME"), hostname=platform.node(),
                                    partition=os.environ.get("SLURM_JOB_PARTITION"),
                                    cpus=os.environ.get("SLURM_CPUS_PER_TASK")),
                    environment=environment_snapshot(),
                    review=dict(review_sha256=review_files[1], gate_sha256=review_files[3]) if review_files else None)
    manifest_path = run_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--phase", choices=("diagnostic", "train", "audit"), required=True)
    for flag in ("source-dir", "raw-root", "records", "split", "stage-dir", "run-dir"):
        prepare.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("config", "weights", "review"):
        prepare.add_argument(f"--{flag}", type=Path)
    prepare.add_argument("--legacy-root", type=Path, default=Path("/nonexistent-anchor-legacy"))
    prepare.add_argument("--weights-sha256")
    check = subcommands.add_parser("check-gate")
    check.add_argument("--run-dir", type=Path, required=True)
    stack = subcommands.add_parser("check-stack")
    stack.add_argument("--report", type=Path, required=True)
    audit = subcommands.add_parser("check-audit")
    audit.add_argument("--run-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "check-gate":
        run = args["run_dir"]
        manifest = json.loads((run / "input_manifest.json").read_text())
        passing_gate(json.loads((run / "pilot/gate.json").read_text()), manifest["runtime_identities"])
        initialization = {}
        for phase in ("overfit", "smoke"):
            path = run / "pilot" / phase / "train.json"
            if path.is_file():
                initialization[phase] = json.loads(path.read_text()).get("initialization", {})
        if initialization:
            (run / "initialization-provenance.json").write_text(json.dumps(initialization, indent=2, sort_keys=True) + "\n")
    elif command == "check-stack":
        check_stack_report(args["report"])
    elif command == "check-audit":
        check_audit_health(args["run_dir"])
    else:
        manifest = prepare_job(**args)
        print(json.dumps(dict(run_dir=manifest["paths"]["run_dir"], provenance=manifest["provenance"])))


if __name__ == "__main__":
    main()
