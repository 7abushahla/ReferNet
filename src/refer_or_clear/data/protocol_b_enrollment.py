"""Explicit Protocol B support authorization and unchanged bank geometry."""
from __future__ import annotations

import copy
from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path

import cv2
import pandas as pd

from ..protocol_b_contract import (
    MembershipRow, ProtocolBContract, REVIEWED_SPLIT_SHA256, TargetEnrollment,
    _authenticated, _canonical, _require, reserve_target_enrollment,
)
from . import enrollment
from .records import provenance_components
from .references import fold_of
from .splits import FSL_GROUPS, NATIONALITY

SCHEMA_VERSION = "protocol_b_enrollment_v1"
GEOMETRY_VERSION = enrollment.GEOMETRY_VERSION


def _source():
    paths = {"protocol_b_enrollment.py": Path(__file__), "protocol_b_contract.py": Path(__file__).parents[1] / "protocol_b_contract.py"}
    paths.update({name: Path(__file__).with_name(name)
                  for name in ("enrollment.py", "records.py", "rectify.py", "references.py", "register.py")})
    return {"implementation_sha256": enrollment._file_hash(Path(__file__)),
            "helpers": {name: enrollment._file_hash(path) for name, path in paths.items()}}


def make_protocol_b_enrollment_manifest(records_csv: Path, contract: ProtocolBContract,
                                        target_enrollment: TargetEnrollment, raw_root: Path) -> dict:
    """Authorize metadata only; raw bytes and genuine layouts are read at build time."""
    _require(isinstance(contract, ProtocolBContract) and isinstance(target_enrollment, TargetEnrollment),
             "explicit Protocol B contract and target enrollment required")
    _require(type(contract.group) is int and contract.group in REVIEWED_SPLIT_SHA256,
             "invalid Protocol B group")
    _require(contract.group_name == FSL_GROUPS[contract.group]
             and contract.split_sha256 == REVIEWED_SPLIT_SHA256[contract.group], "reviewed group/split binding mismatch")
    targets = tuple(sorted(NATIONALITY[name] for name in contract.group_name.split("_")))
    sources = tuple(sorted(set(NATIONALITY.values()) - set(targets)))
    _require(contract.source_designs == sources and contract.target_designs == targets, "nationality contract mismatch")
    target = target_enrollment.as_dict()
    _require(_canonical(target) == _canonical(reserve_target_enrollment(contract)), "target enrollment reservation mismatch")
    _require(isinstance(target_enrollment.manifest_sha256, str) and len(target_enrollment.manifest_sha256) == 64
             and all(character in "0123456789abcdef" for character in target_enrollment.manifest_sha256),
             "target enrollment manifest hash required")
    records_path = Path(records_csv).resolve(strict=True)
    content = _authenticated(records_path, contract.records_sha256, "records")
    records = pd.read_csv(BytesIO(content), keep_default_na=False, dtype=str)
    required = {"image_id", "identity_id", "component_id", "design_id", "label", "dataset", "source_id", "donor_id",
                "path", "annotation", "width", "height"}
    _require(required <= set(records) and records.image_id.is_unique, "missing/duplicate records metadata")
    _require(records.identity_id.str.strip().ne("").all() and records.component_id.str.strip().ne("").all(), "empty provenance identity")
    identities = set(records.identity_id)
    for field in ("source_id", "donor_id"):
        _require(set(records[field]) - {""} <= identities, f"unknown {field} identity token")
    graph = provenance_components(records)
    _require((graph.groupby(records.component_id).nunique() == 1).all()
             and (records.component_id.groupby(graph).nunique() == 1).all(), "transitive provenance component mismatch")
    members = contract.source_train + contract.source_val + contract.target_test
    _require(len({row.image_id for row in members}) == len(members)
             and set(records.loc[records.dataset == "sidtd", "image_id"]) == {row.image_id for row in members},
             "complete classifier membership mismatch")
    indexed = records.set_index("image_id")
    for row in members:
        metadata = indexed.loc[row.image_id]
        _require(metadata.dataset == "sidtd" and metadata.label in ("0", "1")
                 and type(row.label) is int and str(row.label) == metadata.label
                 and all(metadata[field] == getattr(row, field) for field in ("identity_id", "component_id", "design_id")),
                 "records/contract membership mismatch")
    query_ids = set(target_enrollment.query_ids)
    forbidden = {row.component_id for row in contract.source_val}
    forbidden.update(row.component_id for row in contract.target_test if row.image_id in query_ids)
    source_support = tuple(row for row in contract.source_train if row.label == 0 and row.component_id not in forbidden)
    target_support = tuple(row for row in contract.target_test if row.image_id in target_enrollment.support_ids)
    root = Path(raw_root).resolve(strict=True)
    _require(root.is_dir(), "raw_root must be a directory")
    contributors = {}
    for row in sorted(source_support + target_support, key=lambda member: member.image_id):
        metadata = indexed.loc[row.image_id].to_dict()
        metadata["image_id"] = row.image_id
        _require(metadata["annotation"] and metadata["label"] == "0", "support requires genuine layout metadata")
        enrollment._design_path(row.design_id)
        for field in ("path", "annotation"):
            enrollment._relative_path(root, metadata[field], "sidtd")
        for field in ("width", "height"):
            try:
                value = float(metadata[field])
            except (TypeError, ValueError) as error:
                raise ValueError("invalid native dimensions") from error
            _require(math.isfinite(value) and value > 0 and value.is_integer(), "invalid native dimensions")
            metadata[field] = int(value)
        contributors[row.image_id] = {key: metadata[key] for key in (
            "image_id", "dataset", "design_id", "identity_id", "component_id", "path", "annotation", "width", "height")}
    banks = {}
    for design in sources + targets:
        support = source_support if design in sources else target_support
        names = enrollment.BANKS if design in sources else ("enrollment",)
        banks[design] = {}
        for name in names:
            selected = tuple(row for row in support if row.design_id == design
                             and (name == "enrollment" or fold_of(row.component_id) == int(name[-1])))
            _require(selected, f"{design}/{name}: empty authorized support")
            banks[design][name] = dict(protocol="B", group=contract.group, design_id=design, bank=name,
                                      geometry_version=GEOMETRY_VERSION,
                                      support_ids=sorted(row.image_id for row in selected),
                                      component_ids=sorted({row.component_id for row in selected}))
    return dict(schema_version=SCHEMA_VERSION, protocol="B", group=contract.group, group_name=contract.group_name,
                geometry_version=GEOMETRY_VERSION, canvas_hw=[512, 800], raw_root=str(root),
                records_authority=dict(path=str(records_path), sha256=contract.records_sha256),
                official_split_sha256=contract.split_sha256, contract=json.loads(_canonical(asdict(contract))),
                target_enrollment=target, target_enrollment_sha256=target_enrollment.manifest_sha256,
                forbidden_components=sorted(forbidden), source_eligible_ids=sorted(row.image_id for row in source_support),
                target_support_ids=sorted(row.image_id for row in target_support), eligible_ids=sorted(contributors),
                contributors=contributors, banks=banks, source=_source())


def _validate_manifest(manifest, raw_root):
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == SCHEMA_VERSION
             and manifest.get("protocol") == "B", "explicit Protocol B enrollment manifest required")
    try:
        binding = copy.deepcopy(manifest["contract"])
        for role in ("source_train", "source_val", "target_test"):
            binding[role] = tuple(MembershipRow(**row) for row in binding[role])
        for field in ("source_designs", "target_designs"):
            binding[field] = tuple(binding[field])
        contract = ProtocolBContract(**binding)
        target = manifest["target_enrollment"]
        reservation = TargetEnrollment(manifest["target_enrollment_sha256"], tuple(target["support_ids"]),
                                       tuple(target["query_ids"]), tuple(target["excluded_ids"]), _canonical(target))
        expected = make_protocol_b_enrollment_manifest(Path(manifest["records_authority"]["path"]),
                                                       contract, reservation, raw_root)
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete Protocol B metadata manifest") from error
    _require(_canonical(manifest) == _canonical(expected), "Protocol B manifest differs from authenticated metadata")


def build_protocol_b_enrollment_products(manifest: dict, raw_root: Path, output_root: Path) -> dict:
    """CPU bank build from authorized contributors only, with original 512x800 geometry."""
    root, output = Path(raw_root).resolve(strict=True), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    _require(not output.is_relative_to(root) and not root.is_relative_to(output), "output must be separate from raw_root")
    _validate_manifest(manifest, root)
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    built, archives = {}, {}
    try:
        for design, banks in sorted(manifest["banks"].items()):
            built[design] = {}
            for name, selected in sorted(banks.items()):
                directory = output / f"group{manifest['group']}" / enrollment._design_path(design) / name
                seed_slot = dict(selected, fold=manifest["group"])
                bank = enrollment._build_bank(manifest, seed_slot, root, directory, None, archives)
                bank.pop("fold")
                geometry = json.loads(Path(bank["products"]["geometry"]["path"]).read_text())
                geometry.pop("fold")
                geometry.update(schema_version=SCHEMA_VERSION, protocol="B", group=manifest["group"],
                                target_enrollment_sha256=manifest["target_enrollment_sha256"])
                geometry["registration"]["seed_slot_meaning"] = "protocol_b_group"
                bank["products"]["geometry"] = enrollment._write_json(directory / "geometry.json", geometry)
                dependency = dict(geometry, output_sha256={key: item["sha256"] for key, item in bank["products"].items()
                                                          if key != "manifest"})
                bank["products"]["manifest"] = enrollment._write_json(directory / "manifest.json", dependency)
                built[design][name] = bank
        result = copy.deepcopy(manifest)
        result.update(banks=built, output_root=str(output), archive_metadata=archives,
                      counts=dict(designs=len(built), banks=sum(len(banks) for banks in built.values()),
                                  eligible=len(manifest["eligible_ids"])))
        result["contributors"] = {identifier: contributor for banks in built.values()
                                  for identifier, contributor in banks["enrollment"]["contributors"].items()}
        enrollment._write_json(output / "archive_metadata.json", dict(purpose="archival_only_not_bank_dependency",
                                                                    annotation_files=archives))
        enrollment._write_json(output / "manifest.json", result)
        return result
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)


def resolve_protocol_b_bank(manifest: dict, design_id: str, role: str,
                             training_component: str | None = None) -> dict:
    """Source train cross-fits; source val and target test use fixed full banks."""
    _require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("protocol") == "B"
             and manifest.get("geometry_version") == GEOMETRY_VERSION
             and type(manifest.get("group")) is int and manifest["group"] in REVIEWED_SPLIT_SHA256,
             "unsupported Protocol B enrollment schema/group")
    contract = manifest["contract"]
    if role == "train":
        _require(design_id in contract["source_designs"] and isinstance(training_component, str)
                 and training_component, "source training component required")
        _require(any(row["design_id"] == design_id and row["component_id"] == training_component
                     for row in contract["source_train"]), "foreign source training component")
        name = f"train_half{1 - fold_of(training_component)}"
    else:
        _require(training_component is None, "evaluation forbids query component provenance")
        _require((role == "val" and design_id in contract["source_designs"])
                 or (role in ("test", "evaluation") and design_id in contract["target_designs"]),
                 "invalid design/role for Protocol B")
        name = "enrollment"
    try:
        selected = manifest["banks"][design_id][name]
    except KeyError as error:
        raise ValueError("missing Protocol B bank; no global fallback") from error
    _require(selected.get("protocol") == "B" and type(selected.get("group")) is int
             and selected["group"] == manifest["group"] and selected.get("design_id") == design_id
             and selected.get("bank") == name and selected.get("geometry_version") == GEOMETRY_VERSION
             and selected.get("support_ids"), "inconsistent Protocol B bank")
    if design_id in contract["source_designs"]:
        members = [row for row in contract["source_train"] if row["label"] == 0
                   and row["component_id"] not in manifest["forbidden_components"]]
    else:
        members = [row for row in contract["target_test"] if row["label"] == 0
                   and row["image_id"] in manifest["target_enrollment"]["support_ids"]]
    members = [row for row in members if row["design_id"] == design_id
               and (name == "enrollment" or fold_of(row["component_id"]) == int(name[-1]))]
    _require(selected["support_ids"] == sorted(row["image_id"] for row in members)
             and selected.get("component_ids") == sorted({row["component_id"] for row in members}),
             "bank membership differs from authorized support")
    if role == "train":
        _require(training_component not in selected["component_ids"], "query component in training support")
    return copy.deepcopy(selected)
