"""Stage immutable inputs for the bounded direct-reference GPU diagnostic."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


ENROLLMENT_SHA256 = "7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8"
LEVIR_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"


def load_script(path):
    spec = importlib.util.spec_from_file_location("_direct_reference_staging_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    return load_script(Path(__file__).resolve().with_name("anchor_job_manifest.py"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_path(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "absolute path without traversal required")
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), f"symlink path forbidden: {parent}")
    return path


def authenticate(binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "path/hash authority required")
    require(isinstance(binding["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]), "invalid SHA256")
    path = absolute_path(binding["path"])
    require(path.is_file(), f"authority file missing: {path}")
    require(helpers().file_sha256(path) == binding["sha256"], f"authority SHA256 mismatch: {path}")
    return path


def read_json(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")

    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "JSON object required")
    return value


def source_digest(source):
    common = helpers()
    return common.object_sha256(common.source_entries(absolute_path(source)))


def validate_allocation():
    require(os.environ.get("SLURM_JOB_PARTITION") == "gpu", "GPU allocation required")
    require(re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")), "Slurm job ID required")
    node = os.environ.get("SLURMD_NODENAME", "").lower()
    require(node and not any(name in node for name in ("login", "head", "frontend", "lgn")), "compute node required")


def write_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o444)
    return {"path": str(path), "sha256": helpers().file_sha256(path)}


def prepare_job(*, source_dir, source_sha256, request, request_sha256, stage_dir, run_dir):
    validate_allocation()
    common = helpers()
    source = absolute_path(source_dir)
    source_files = common.source_entries(source)
    require(common.object_sha256(source_files) == source_sha256, "source snapshot SHA256 mismatch")
    request_binding = {"path": str(request), "sha256": request_sha256}
    request_path = authenticate(request_binding)
    settings = read_json(request_path)
    require(settings.get("schema_version") == "direct_reference_diagnostic_request_v1"
            and settings.get("source_sha256") == source_sha256, "diagnostic request/source contract mismatch")
    required = ("config", "records", "split", "weights", "enrollment")
    authorities = {name: authenticate(settings[name]) for name in required}
    require(settings["weights"]["sha256"] == LEVIR_SHA256, "official LEVIR weights required")
    require(settings["enrollment"]["sha256"] == ENROLLMENT_SHA256, "reviewed fold-zero enrollment required")
    require(authorities["config"].is_relative_to(source / "configs"), "configuration must belong to source snapshot")
    raw = absolute_path(settings["raw_root"])
    products = absolute_path(settings["product_root"])
    enrollment = read_json(authorities["enrollment"])
    original_products = absolute_path(enrollment["output_root"])
    verifier = load_script(source / "scripts/verify_anchor_enrollment_products.py")
    integrity = verifier.verify_products(authorities["enrollment"], ENROLLMENT_SHA256, output_root=products)
    require(integrity.get("passed") is True, "enrollment product integrity failed")
    rows = common.selected_rows(authorities["records"], authorities["split"], "R")
    counts = {role: sum(row["role"] == role for row in rows) for role in ("train", "val")}
    require(counts == {"train": 1778, "val": 222}, "official fold-zero train/validation counts required")
    stage, run = absolute_path(stage_dir), absolute_path(run_dir)
    protected = (source, raw, products, original_products, request_path, *authorities.values())
    for destination in (stage, run):
        if destination.exists():
            raise FileExistsError(f"new output directory required: {destination}")
        require(all(not destination.is_relative_to(root) and not root.is_relative_to(destination) for root in protected),
                "output must be isolated from inputs")
    require(not stage.is_relative_to(run) and not run.is_relative_to(stage), "scratch and results must be separate")
    query_files = common.data_entries(rows, "R", raw, raw)
    product_files = []
    for banks in enrollment["banks"].values():
        for bank in banks.values():
            for entry in bank["products"].values():
                declared = absolute_path(entry["path"])
                require(declared.is_relative_to(original_products), "product outside authorized bank root")
                relative = declared.relative_to(original_products)
                actual = common.input_file(products, relative)
                require(common.file_sha256(actual) == entry["sha256"], "enrollment product hash changed")
                product_files.append({"path": relative.as_posix(), "sha256": entry["sha256"]})
    stage.mkdir(parents=True)
    run.mkdir(parents=True)
    for directory in ("raw", "code", "inputs", "enrollment", "tmp"):
        (stage / directory).mkdir()
    for entry in source_files:
        common.copy_verified(source / entry["path"], stage / "code" / entry["path"], entry["sha256"])
    require(source_digest(source) == source_sha256, "source snapshot changed during copy")
    require(source_digest(stage / "code") == source_sha256, "frozen source snapshot SHA256 mismatch")
    filenames = {"config": "config.yaml", "records": "records.csv", "split": "split.csv",
                 "weights": "weights.pth", "enrollment": "enrollment.json"}
    frozen = {}
    for name, path in authorities.items():
        target = stage / "inputs" / filenames[name]
        common.copy_verified(path, target, settings[name]["sha256"])
        frozen[name] = {"path": str(target), "sha256": settings[name]["sha256"]}
    common.copy_verified(request_path, stage / "inputs/request.json", request_sha256)
    for entry in query_files:
        common.copy_verified(common.input_file(raw, entry["path"]), stage / "raw" / entry["path"], entry["sha256"])
    for entry in product_files:
        common.copy_verified(common.input_file(products, entry["path"]), stage / "enrollment" / entry["path"], entry["sha256"])
    relocation = write_json(stage / "inputs/product-root.json", {
        "schema_version": "direct_reference_product_root_v1", "original_root": str(original_products),
        "relocated_root": str(stage / "enrollment"), "enrollment_manifest_sha256": ENROLLMENT_SHA256})
    require(verifier.verify_products(frozen["enrollment"]["path"], ENROLLMENT_SHA256,
                                    output_root=stage / "enrollment").get("passed") is True,
            "frozen enrollment product integrity failed")
    result = {"schema_version": "direct_reference_job_v1", "diagnostic_only": True,
              "request": {"path": str(stage / "inputs/request.json"), "sha256": request_sha256},
              "source_sha256": source_sha256, "source_files": source_files,
              "authorities": frozen, "product_root_authority": relocation,
              "membership_counts": counts, "membership": {role: [row["image_id"] for row in rows if row["role"] == role]
                                                          for role in ("train", "val")},
              "query_files": query_files, "product_files": product_files,
              "paths": {"source_dir": str(stage / "code"), "raw_root": str(stage / "raw"),
                        "product_root": str(stage / "enrollment"), "stage_dir": str(stage), "run_dir": str(run)},
              "allocation": {name: os.environ.get(name) for name in
                             ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME")}}
    write_json(run / "input_manifest.json", result)
    for directory in ("raw", "code", "inputs", "enrollment"):
        common.freeze_directories(stage / directory)
    return result


def check_stack_report(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases = list(root.iter("testcase"))
    require(suites and len(cases) == 1 and sum(int(suite.get("tests", "0")) for suite in suites) == 1,
            "one actual stack test required")
    require(cases[0].get("name") == "test_actual_direct_reference_gpu_integration"
            and cases[0].get("classname", "").split(".")[-1] == "test_direct_reference_model",
            "named direct-reference actual stack test required")
    require(not any(list(cases[0].iter(field)) for field in ("failure", "error", "skipped")),
            "actual stack testcase must pass without skips")
    require(all(int(suite.get(field, "0")) == 0 for suite in suites for field in ("failures", "errors", "skipped")),
            "actual stack test must pass without skips")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("source-dir", "source-sha256", "request", "request-sha256", "stage-dir", "run-dir"):
        prepare.add_argument("--" + name, required=True)
    check = commands.add_parser("check-stack")
    check.add_argument("--report", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "check-stack":
        return check_stack_report(args["report"])
    return prepare_job(**args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
