"""Pinned A-source/C-headline metadata and approved recorded-lineage reservation.

This is evaluation-only model transfer, not legacy C training. No checkpoints,
annotations, or image bytes are opened. A source train/validation component
overlap is retained as published, not silently replaced by B's disjoint split.
Reservation excludes all matching recorded identities/components from the
599-image baseline. Missing donor metadata cannot prove donor-disjointness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path

import pandas as pd

from .data.records import provenance_components

RECORDS_SHA256 = "afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276"
SOURCE_SPLIT_SHA256 = "985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9"
TARGET_SPLIT_SHA256 = "816ef2eb148eaa34f65ff198fbe6854119564565454dd4afaa243990dd2198a9"
SOURCE_COUNTS = {"train": 1778, "val": 222, "test": 222}
TARGET_COUNT = 599
TARGET_DESIGNS = ("fantasyid/chinese2", "fantasyid/netherland", "fantasyid/singapore")
_spec = importlib.util.spec_from_file_location("protocol_c_file_verifier", Path(__file__).resolve().parents[2] / "scripts/verify_anchor_enrollment_products.py")
_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_verifier)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _path(value):
    _require(isinstance(value, (str, Path)), "path required")
    path = _verifier._absolute(Path(value).absolute())
    _verifier._no_symlinks(path)
    return path


def _authenticated(path, expected_sha256, name):
    _require(_verifier._sha256_value(expected_sha256), f"explicit {name} SHA256 required")
    observed, payload = _verifier._read_file(_path(path), capture=True)
    _require(observed == expected_sha256, f"{name} SHA256 hash mismatch")
    return payload


@dataclass(frozen=True)
class MembershipRow:
    image_id: str
    identity_id: str
    component_id: str
    design_id: str
    label: int


@dataclass(frozen=True)
class ProtocolCContract:
    records_path: str
    source_split_path: str
    target_split_path: str
    records_sha256: str
    source_split_sha256: str
    target_split_sha256: str
    source_train: tuple[MembershipRow, ...]
    source_val: tuple[MembershipRow, ...]
    target_test: tuple[MembershipRow, ...]
    target_designs: tuple[str, ...]
    donor_unknown_altered_count: int


@dataclass(frozen=True)
class TargetEnrollment:
    path: str
    manifest_sha256: str
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    _serialized: str

    def as_dict(self):
        return json.loads(self._serialized)


def _frames(records_csv, source_split_csv, target_split_csv, records_sha256):
    _require(records_sha256 == RECORDS_SHA256, "unapproved records pin")
    contents = [_authenticated(path, pin, name) for path, pin, name in (
        (records_csv, RECORDS_SHA256, "records"), (source_split_csv, SOURCE_SPLIT_SHA256, "source split"),
        (target_split_csv, TARGET_SPLIT_SHA256, "target split"))]
    records, source, target = [pd.read_csv(BytesIO(payload), keep_default_na=False, dtype=str) for payload in contents]
    required = {"image_id", "identity_id", "component_id", "design_id", "label", "dataset", "source_id", "donor_id"}
    _require(required <= set(records), "records provenance columns missing")
    for frame in (records, source, target):
        _require({"image_id", "component_id", "design_id", "label"} <= set(frame), "membership columns missing")
        _require(len(frame) and frame.image_id.is_unique, "duplicate or empty membership")
        for field in ("image_id", "component_id", "design_id"):
            _require(frame[field].str.strip().ne("").all(), f"empty {field}")
        _require(frame.label.isin(("0", "1")).all(), "labels must be binary integers")
    _require(records.identity_id.str.strip().ne("").all(), "empty identity_id")
    for field in ("source_id", "donor_id"):
        _require(set(records[field]) - {""} <= set(records.identity_id), f"unknown {field} identity token")
    graph = provenance_components(records)
    _require((graph.groupby(records.component_id).nunique() == 1).all()
             and (records.component_id.groupby(graph).nunique() == 1).all(), "transitive recorded component mismatch")
    _require({"protocol", "role", "ref_group"} <= set(source)
             and source.protocol.eq("A").all() and source.ref_group.eq("it0").all(), "official A fold-zero source required")
    _require(source.role.value_counts().to_dict() == SOURCE_COUNTS, "source official membership counts mismatch")
    _require(set(source.image_id) == set(records.loc[records.dataset == "sidtd", "image_id"]), "complete A membership mismatch")
    _require({"protocol", "role", "subset"} <= set(target) and target.protocol.eq("C").all()
             and set(target.role) <= {"train", "val", "test", "reference_only"}, "legacy C target membership required")
    indexed = records.set_index("image_id")
    joins = []
    for frame in (source, target):
        _require(set(frame.image_id) <= set(indexed.index), "split ID missing from records")
        joined = frame.join(indexed[["identity_id", "component_id", "design_id", "label", "dataset", "donor_id"]],
                            on="image_id", rsuffix="_record", validate="one_to_one")
        for field in ("component_id", "design_id", "label"):
            _require(joined[field].eq(joined[f"{field}_record"]).all(), f"records/split {field} mismatch")
        joins.append(joined)
    return records, joins[0], joins[1]


def load_protocol_c_contract(records_csv, source_split_csv, target_split_csv, *, records_sha256=RECORDS_SHA256):
    """Authenticate original CSV pins, with no target outcome or pixel access."""
    _, source, target = _frames(records_csv, source_split_csv, target_split_csv, records_sha256)
    target = target.loc[(target.role == "test") & (target.subset == "heldout")]
    _require(len(target) == TARGET_COUNT and set(target.design_id) == set(TARGET_DESIGNS)
             and target.dataset.eq("fantasyid").all(), "exact C headline membership required")
    def members(frame):
        return tuple(MembershipRow(row.image_id, row.identity_id, row.component_id, row.design_id, int(row.label))
                     for row in frame.sort_values("image_id").itertuples())
    train, val, queries = members(source.loc[source.role == "train"]), members(source.loc[source.role == "val"]), members(target)
    for field in ("image_id", "identity_id", "component_id"):
        _require(not {getattr(row, field) for row in train + val} & {getattr(row, field) for row in queries}, "source/target provenance overlap")
    return ProtocolCContract(str(_path(records_csv)), str(_path(source_split_csv)), str(_path(target_split_csv)),
        RECORDS_SHA256, SOURCE_SPLIT_SHA256, TARGET_SPLIT_SHA256, train, val, queries, TARGET_DESIGNS,
        int(((target.label == "1") & (target.donor_id == "")).sum()))


def _validate_contract(contract):
    _require(isinstance(contract, ProtocolCContract), "explicit C contract required")
    expected = load_protocol_c_contract(contract.records_path, contract.source_split_path, contract.target_split_path,
                                       records_sha256=contract.records_sha256)
    _require(_canonical(asdict(contract)) == _canonical(asdict(expected)), "C contract differs from pinned membership")


def reserve_target_enrollment(contract):
    """Deterministic recorded-component reservation, never target training."""
    _validate_contract(contract)
    reserved = set()
    for design in contract.target_designs:
        candidates = {row.component_id for row in contract.target_test if row.design_id == design and row.label == 0}
        count = max(2, math.ceil(len(candidates) * .2))
        _require(len(candidates) >= count, f"{design}: insufficient eligible genuine components")
        ranked = sorted(candidates, key=lambda component: (
            hashlib.sha256(f"protocol_c_target_support_v1:42:{design}:{component}".encode()).hexdigest(), component))
        reserved.update(ranked[:count])
    identities = {row.identity_id for row in contract.target_test if row.component_id in reserved and row.label == 0}
    matched = lambda row: row.component_id in reserved or row.identity_id in identities
    support = tuple(row for row in contract.target_test if matched(row) and row.label == 0)
    excluded = tuple(row for row in contract.target_test if matched(row) and row.label == 1)
    queries = tuple(row for row in contract.target_test if not matched(row))
    for design in contract.target_designs:
        _require(any(row.design_id == design for row in queries), f"{design}: no remaining query population")
        _require(len({row.component_id for row in support if row.design_id == design}) >= 2, "insufficient support components")
    for field in ("identity_id", "component_id"):
        _require(not {getattr(row, field) for row in support} & {getattr(row, field) for row in queries}, "support/query recorded lineage overlap")
    roles = {row.image_id: role for rows, role in ((support, "target_support"), (queries, "target_query"), (excluded, "target_excluded")) for row in rows}
    return dict(schema_version="protocol_c_target_enrollment_v1", protocol="C", deployment="locked_a_selected_models_no_target_training",
        records_sha256=contract.records_sha256, source_split_sha256=contract.source_split_sha256, target_split_sha256=contract.target_split_sha256,
        baseline_ids=sorted(row.image_id for row in contract.target_test), support_ids=sorted(row.image_id for row in support),
        query_ids=sorted(row.image_id for row in queries), excluded_ids=sorted(row.image_id for row in excluded), roles=dict(sorted(roles.items())),
        excluded_reasons={row.image_id: "reserved_recorded_lineage_derived_attack" for row in excluded},
        selection=dict(algorithm="sha256_component_rank_v1", namespace="protocol_c_target_support_v1", seed=42, fraction=.2, minimum_support_components=2, rounding="ceil"),
        published_target_count=len(contract.target_test), query_count=len(queries), published_test_population_preserved=False,
        lineage=dict(scope="recorded_identity_component_only", donor_unknown_altered_count=contract.donor_unknown_altered_count,
                     complete_donor_disjointness_claimed=False))


def load_target_enrollment(contract, manifest_path, *, expected_sha256):
    document = _verifier._json(_authenticated(manifest_path, expected_sha256, "target enrollment"))
    expected = reserve_target_enrollment(contract)
    _require(_canonical(document) == _canonical(expected), "C target enrollment differs from deterministic reservation")
    return TargetEnrollment(str(_path(manifest_path)), expected_sha256, tuple(document["support_ids"]), tuple(document["query_ids"]),
                            tuple(document["excluded_ids"]), _canonical(document))
