"""Metadata-only Protocol B authority and explicitly reduced target population."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from .data.records import provenance_components
from .data.splits import FSL_GROUPS, NATIONALITY

REVIEWED_SPLIT_SHA256 = MappingProxyType(dict(enumerate((
    "58ea2994181c6bbf4b819ed81b4ae73195912a29611977aea00c4247266a82d1",
    "74aef362c0b6e0e5b77bdf5b8f819b94b73e86240b60ac6a2f8cb98a66565ea9",
    "7de884dd4931132d7651c5042df65c993e66bbc75c12852db1837fd4c98e7834",
    "aa2598ed738ae694d2a4dcd6c00f84a6d7c402dd5b9b6b826912955731c7805d",
    "9879e5979091254cc358949a35bc6f99321152dcbebe47eca0e7c5a68195821e",
    "e6af0efc40ea909a48ab15f7c89ad8c65465647788af62fa9201017b657429cf",
    "f451d14b3f8ba0f3f742ccc4bc36bf6eda695b71709cc16441e0377b051e312f",
    "64d41caab2d0158955b9c6c7f09db9ad3f40fc2fe49443f95ddf59de8a2b7fea",
    "7b713d32b1f29d5b6bf358015192edb43b8d7c7bad92db032a38412693d9c3bf",
    "49012ce73acc55d3e7b402ea1913c6c11f4f8f34c8eb0a823811f75c47f22840",
))))


@dataclass(frozen=True)
class MembershipRow:
    image_id: str
    identity_id: str
    component_id: str
    design_id: str
    label: int


@dataclass(frozen=True)
class ProtocolBContract:
    group: int
    group_name: str
    records_sha256: str
    split_sha256: str
    source_designs: tuple[str, ...]
    target_designs: tuple[str, ...]
    source_train: tuple[MembershipRow, ...]
    source_val: tuple[MembershipRow, ...]
    target_test: tuple[MembershipRow, ...]


@dataclass(frozen=True)
class TrainingAccounting:
    n_train: int
    n_val: int
    n_target_test: int
    batches_per_epoch: int
    images_per_epoch: int
    omitted_per_epoch: int
    total_optimizer_steps: int
    effective_warmup_iters: int


@dataclass(frozen=True)
class TargetEnrollment:
    manifest_sha256: str
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    _serialized: str

    def as_dict(self) -> dict:
        return json.loads(self._serialized)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _authenticated(path, expected_sha256, name):
    _require(path is not None and isinstance(expected_sha256, str)
             and len(expected_sha256) == 64
             and all(character in "0123456789abcdef" for character in expected_sha256),
             f"explicit {name} path and SHA256 required")
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{name} must be a regular metadata file")
    content = path.read_bytes()
    _require(hashlib.sha256(content).hexdigest() == expected_sha256, f"{name} hash mismatch")
    return content


def _disjoint(first, second, description):
    for field in ("image_id", "identity_id", "component_id"):
        _require(not {getattr(row, field) for row in first} & {getattr(row, field) for row in second},
                 f"{description}: overlapping {field}")


def load_protocol_b_contract(records_csv: Path, split_csv: Path, *, group: int,
                             records_sha256: str) -> ProtocolBContract:
    """Read only authenticated CSV bytes, never redraw a split or open pixels."""
    _require(type(group) is int and group in REVIEWED_SPLIT_SHA256, "group must be an integer in 0..9")
    split_content = _authenticated(split_csv, REVIEWED_SPLIT_SHA256[group], "split")
    records_content = _authenticated(records_csv, records_sha256, "records")
    split = pd.read_csv(BytesIO(split_content), keep_default_na=False, dtype=str)
    records = pd.read_csv(BytesIO(records_content), keep_default_na=False, dtype=str)
    record_columns = {"image_id", "identity_id", "component_id", "design_id", "label", "dataset", "source_id", "donor_id"}
    split_columns = {"image_id", "component_id", "design_id", "label", "protocol", "role"}
    _require(record_columns <= set(records) and split_columns <= set(split), "missing membership/provenance columns")
    for frame in (records, split):
        _require(frame.image_id.is_unique and len(frame) > 0, "duplicate or empty membership")
        for field in ("image_id", "component_id", "design_id"):
            _require(frame[field].str.strip().ne("").all(), f"empty {field}")
        _require(frame.label.isin(("0", "1")).all(), "labels must be binary integers")
    _require(records.identity_id.str.strip().ne("").all(), "empty identity_id")
    identities = set(records.identity_id)
    for field in ("source_id", "donor_id"):
        _require(set(records[field]) - {""} <= identities, f"unknown {field} identity token")
    components = provenance_components(records)
    _require((components.groupby(records.component_id).nunique() == 1).all()
             and (records.component_id.groupby(components).nunique() == 1).all(),
             "component IDs differ from transitive identity/source/donor connectivity")
    sidtd = records.loc[records.dataset == "sidtd"].set_index("image_id")
    _require(set(sidtd.index) == set(split.image_id), "records/split membership mismatch")
    _require(split.protocol.eq("B").all() and set(split.role) == {"train", "val", "test"}, "Protocol B roles required")
    joined = split.join(sidtd[["identity_id", "component_id", "design_id", "label"]],
                        on="image_id", rsuffix="_record", validate="one_to_one")
    for field in ("component_id", "design_id", "label"):
        _require(joined[field].eq(joined[f"{field}_record"]).all(), f"records/split {field} mismatch")
    target_designs = tuple(sorted(NATIONALITY[name] for name in FSL_GROUPS[group].split("_")))
    source_designs = tuple(sorted(set(NATIONALITY.values()) - set(target_designs)))
    _require(set(joined.loc[joined.role == "test", "design_id"]) == set(target_designs), "target nationality membership mismatch")
    for role in ("train", "val"):
        _require(set(joined.loc[joined.role == role, "design_id"]) == set(source_designs), "source nationality membership mismatch")
    members = {role: tuple(MembershipRow(row.image_id, row.identity_id, row.component_id,
                                        row.design_id, int(row.label))
                           for row in joined.loc[joined.role == role].sort_values("image_id").itertuples())
               for role in ("train", "val", "test")}
    _disjoint(members["train"], members["val"], "source train/val")
    _disjoint(members["train"] + members["val"], members["test"], "source/target")
    return ProtocolBContract(group, FSL_GROUPS[group], records_sha256, REVIEWED_SPLIT_SHA256[group],
                             source_designs, target_designs, members["train"], members["val"], members["test"])


def training_accounting(contract: ProtocolBContract, recipe: dict) -> TrainingAccounting:
    """Current 100-epoch classification recipe, with group-derived update counts."""
    fixed = dict(epochs=100, batch_size=8, eval_batch_size=1, accumulation_steps=1,
                 drop_last=True, eval_drop_last=False, seed=42)
    _require(isinstance(recipe, dict) and all(type(recipe.get(key)) is type(value)
             and recipe[key] == value for key, value in fixed.items()), "unsupported Protocol B recipe")
    scheduler = recipe.get("scheduler")
    _require(isinstance(scheduler, dict) and scheduler.get("name") == "warmup_cosine"
             and type(scheduler.get("warmup_iters")) is int and scheduler["warmup_iters"] == 1000
             and type(scheduler.get("warmup_cap_fraction")) in (int, float)
             and scheduler["warmup_cap_fraction"] == 0.1, "unsupported scheduler recipe")
    batches = len(contract.source_train) // recipe["batch_size"]
    _require(batches > 0 and len(contract.source_val) > 0, "empty source training/validation population")
    images = batches * recipe["batch_size"]
    total = batches * recipe["epochs"]
    return TrainingAccounting(len(contract.source_train), len(contract.source_val), len(contract.target_test),
                              batches, images, len(contract.source_train) - images, total,
                              min(scheduler["warmup_iters"], int(scheduler["warmup_cap_fraction"] * total)))


def make_protocol_b_scheduler(optimizer, contract: ProtocolBContract, recipe: dict):
    from .pilots import pilot_scheduler

    accounting = training_accounting(contract, recipe)
    return pilot_scheduler(optimizer, "D_running_pilot", accounting.total_optimizer_steps)


def reserve_target_enrollment(contract: ProtocolBContract) -> dict:
    """Serialize the approved fixed reservation, not an allocation authorization."""
    source = contract.source_train + contract.source_val
    _disjoint(source, contract.target_test, "source/target")
    reserved = set()
    for design in contract.target_designs:
        candidates = {row.component_id for row in contract.target_test if row.design_id == design and row.label == 0}
        count = max(2, math.ceil(len(candidates) * 0.2))
        _require(len(candidates) >= count, f"{design}: fewer than two eligible genuine components")
        ranked = sorted(candidates, key=lambda component: (
            hashlib.sha256(f"protocol_b_target_support_v1:42:{design}:{component}".encode()).hexdigest(), component))
        reserved.update(ranked[:count])
    support = tuple(row for row in contract.target_test if row.component_id in reserved and row.label == 0)
    excluded = tuple(row for row in contract.target_test if row.component_id in reserved and row.label != 0)
    queries = tuple(row for row in contract.target_test if row.component_id not in reserved)
    _disjoint(support, queries, "target support/query")
    for design in contract.target_designs:
        _require(any(row.design_id == design for row in queries), f"{design}: no remaining query population")
        _require(len({row.component_id for row in support if row.design_id == design}) >= 2,
                 f"{design}: insufficient reserved support")
    roles = {row.image_id: role for members, role in ((contract.source_train, "source_train"),
             (contract.source_val, "source_val"), (support, "target_support"),
             (queries, "target_query"), (excluded, "target_excluded")) for row in members}
    reasons = {row.image_id: "reserved_genuine_support" for row in support}
    reasons.update({row.image_id: "reserved_component_derived_attack" for row in excluded})
    return dict(schema_version="protocol_b_target_enrollment_v1", group=contract.group,
                group_name=contract.group_name, records_sha256=contract.records_sha256,
                split_sha256=contract.split_sha256,
                selection=dict(algorithm="sha256_component_rank_v1", seed=42, fraction=0.2,
                               minimum_support_components=2, rounding="ceil"),
                support_ids=sorted(row.image_id for row in support), query_ids=sorted(row.image_id for row in queries),
                excluded_ids=sorted(row.image_id for row in excluded), roles=dict(sorted(roles.items())),
                excluded_reasons=dict(sorted(reasons.items())), published_target_count=len(contract.target_test),
                query_count=len(queries), published_test_population_preserved=False)


def _canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_target_enrollment(contract: ProtocolBContract, manifest_path: Path, *,
                           expected_sha256: str) -> TargetEnrollment:
    """Require externally bound explicit membership before any target inference."""
    content = _authenticated(manifest_path, expected_sha256, "target enrollment manifest")
    manifest = json.loads(content)
    _require(isinstance(manifest, dict), "target enrollment manifest must be an object")
    expected = reserve_target_enrollment(contract)
    _require(_canonical(manifest) == _canonical(expected), "target enrollment manifest differs from frozen reservation/identities")
    return TargetEnrollment(expected_sha256, tuple(manifest["support_ids"]), tuple(manifest["query_ids"]),
                            tuple(manifest["excluded_ids"]), _canonical(manifest))
