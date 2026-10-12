"""C-only CPU target support, reusing original bank geometry without fake folds.

Metadata authorization opens only pinned CSV/reservation bytes. Product building
opens only authorized genuine FantasyID contributors and their layout JSON.
Final products live at groupC/<design>/enrollment and use C schemas throughout.
The original helper's integer seed slot is zero, explicitly not an A/B fold.
No checkpoint authority is required here; evaluation is a separate transition.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import cv2
import pandas as pd

from ..protocol_c_contract import (
    ProtocolCContract, TargetEnrollment, _authenticated, _canonical, _path,
    _require, _validate_contract, load_protocol_c_contract, load_target_enrollment,
)
from . import enrollment

SCHEMA_VERSION = "protocol_c_enrollment_v1"
GEOMETRY_VERSION = enrollment.GEOMETRY_VERSION
PRODUCTS = {"target", "target_coverage", "target_valid", "median", "coverage", "valid", "zone", "geometry", "manifest"}


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _source():
    paths = {"protocol_c_enrollment.py": Path(__file__), "protocol_c_contract.py": Path(__file__).parents[1] / "protocol_c_contract.py"}
    paths.update({name: Path(__file__).with_name(name) for name in ("enrollment.py", "rectify.py", "references.py", "register.py", "records.py")})
    return {"implementation_sha256": enrollment._file_hash(Path(__file__)),
            "helpers": {name: enrollment._file_hash(path) for name, path in paths.items()}}


def make_protocol_c_enrollment_manifest(contract, target_enrollment, raw_root):
    """Authorize selected genuine support paths, not pixel availability or fits."""
    _validate_contract(contract)
    _require(isinstance(target_enrollment, TargetEnrollment), "explicit C target enrollment required")
    target = load_target_enrollment(contract, target_enrollment.path, expected_sha256=target_enrollment.manifest_sha256)
    _require(_canonical(asdict(target)) == _canonical(asdict(target_enrollment)), "target authority was mutated")
    records = pd.read_csv(BytesIO(_authenticated(contract.records_path, contract.records_sha256, "records")), keep_default_na=False, dtype=str)
    _require({"path", "annotation", "width", "height"} <= set(records), "support filename/native geometry columns missing")
    root = _path(raw_root)
    _require(root.is_dir(), "raw_root must be a directory")
    indexed = records.set_index("image_id")
    contributors = {}
    for identifier in target.support_ids:
        metadata = indexed.loc[identifier].to_dict()
        _require(metadata["dataset"] == "fantasyid" and metadata["label"] == "0" and metadata["annotation"], "support requires genuine FantasyID layout authority")
        enrollment._design_path(metadata["design_id"])
        for field in ("path", "annotation"):
            relative = Path(metadata[field])
            _require(not relative.is_absolute() and relative.parts and relative.parts[0] == "fantasyid"
                     and ".." not in relative.parts and "\\" not in metadata[field], "unsafe support filename")
            _path(root / relative)
        for field in ("width", "height"):
            try:
                value = float(metadata[field])
            except (TypeError, ValueError) as error:
                raise ValueError("invalid support native dimensions") from error
            _require(math.isfinite(value) and value > 0 and value.is_integer(), "invalid support native dimensions")
            metadata[field] = int(value)
        metadata["image_id"] = identifier
        contributors[identifier] = {key: metadata[key] for key in (
            "image_id", "dataset", "design_id", "identity_id", "component_id", "path", "annotation", "width", "height")}
    banks = {}
    for design in contract.target_designs:
        members = [item for item in contributors.values() if item["design_id"] == design]
        _require(members, "empty authorized target support")
        banks[design] = {"enrollment": dict(protocol="C", design_id=design, bank="enrollment", geometry_version=GEOMETRY_VERSION,
            support_ids=sorted(item["image_id"] for item in members), component_ids=sorted({item["component_id"] for item in members}))}
    return dict(schema_version=SCHEMA_VERSION, protocol="C", geometry_version=GEOMETRY_VERSION, canvas_hw=[512, 800],
        deployment="locked_a_selected_models_no_target_training", raw_root=str(root),
        contract=json.loads(_canonical(asdict(contract))), target_enrollment=target.as_dict(),
        target_enrollment_authority=dict(path=target.path, sha256=target.manifest_sha256),
        source_split_sha256=contract.source_split_sha256, target_split_sha256=contract.target_split_sha256,
        records_sha256=contract.records_sha256, eligible_ids=sorted(contributors), contributors=contributors,
        banks=banks, source=_source(), seed_slot=0, seed_slot_meaning="protocol_c_fixed_geometry_slot_not_fold")


def _reload_contract(manifest):
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == SCHEMA_VERSION
             and manifest.get("protocol") == "C", "explicit C enrollment manifest required")
    try:
        declaration = manifest["contract"]
        contract = load_protocol_c_contract(declaration["records_path"], declaration["source_split_path"], declaration["target_split_path"],
                                            records_sha256=declaration["records_sha256"])
        binding = manifest["target_enrollment_authority"]
        target = load_target_enrollment(contract, binding["path"], expected_sha256=binding["sha256"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("incomplete C enrollment authority") from error
    return contract, target


def _validate_manifest(manifest, root):
    contract, target = _reload_contract(manifest)
    expected = make_protocol_c_enrollment_manifest(contract, target, root)
    _require(_canonical(manifest) == _canonical(expected), "C enrollment manifest differs from authenticated metadata")


def build_protocol_c_enrollment_products(manifest, raw_root, output_root):
    """Build three full fixed banks on CPU; no source/query/altered image access."""
    _require(isinstance(manifest, dict), "C metadata manifest required")
    manifest = copy.deepcopy(manifest)
    root, output = _path(raw_root), _path(output_root)
    _require(not output.is_relative_to(root) and not root.is_relative_to(output), "output must not overlap raw_root")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    _validate_manifest(manifest, root)
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    built, archives = {}, {}
    try:
        internal = dict(manifest, official_split_sha256=manifest["target_split_sha256"])
        for design, banks in sorted(manifest["banks"].items()):
            selected = banks["enrollment"]
            directory = output / "groupC" / enrollment._design_path(design) / "enrollment"
            bank = enrollment._build_bank(internal, dict(selected, fold=0), root, directory, None, archives)
            bank.pop("fold")
            geometry_path = Path(bank["products"]["geometry"]["path"])
            geometry = json.loads(geometry_path.read_text())
            geometry.pop("fold")
            geometry.pop("official_split_sha256")
            geometry.update(schema_version=SCHEMA_VERSION, protocol="C", seed_slot=0,
                source_split_sha256=manifest["source_split_sha256"], target_split_sha256=manifest["target_split_sha256"],
                target_enrollment_sha256=manifest["target_enrollment_authority"]["sha256"])
            geometry["registration"].update(seed_slot_meaning=manifest["seed_slot_meaning"],
                seed_derivation="sha256(version,seed_slot,design,bank,stage,image_id) low31")
            bank["products"]["geometry"] = enrollment._write_json(geometry_path, geometry)
            dependency = dict(geometry, output_sha256={name: binding["sha256"] for name, binding in bank["products"].items() if name != "manifest"})
            bank["products"]["manifest"] = enrollment._write_json(directory / "manifest.json", dependency)
            _require(set(bank["products"]) == PRODUCTS, "exact nine C products required")
            built[design] = {"enrollment": bank}
        result = copy.deepcopy(manifest)
        result.update(banks=built, output_root=str(output), authorized_metadata_sha256=_digest(manifest), archive_metadata=archives,
            counts=dict(designs=len(built), banks=len(built), eligible=len(manifest["eligible_ids"]), products=sum(len(item["enrollment"]["products"]) for item in built.values())))
        result["contributors"] = {identifier: item for banks in built.values() for identifier, item in banks["enrollment"]["contributors"].items()}
        enrollment._write_json(output / "archive_metadata.json", dict(purpose="archival_only_not_bank_dependency", annotation_files=archives))
        enrollment._write_json(output / "manifest.json", result)
        return result
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
