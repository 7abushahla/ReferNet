"""Authenticated official A membership without identity-disjoint reinterpretation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from .data.records import provenance_components

RECORDS_SHA256 = "afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276"
REVIEWED_SPLIT_SHA256 = MappingProxyType(dict(enumerate((
    "985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9",
    "c7d108092c7408dc34b19a629bfc92dc401985e9555500e3495e13d66f9f91c7",
    "8ac63c774cb78085c9d66604c6d71825edd7b07c8babbd3854169e0421e4b041",
    "7a28608725cd799221b2f65b0ae330d746eadb6470770e84495786d97684d8e7",
    "d35773f8495bfed2aa5f105eae67f62b40450cd55e15f4bfe038c1d87b8b16a7",
    "ce9f413e18fafb53191b2a4f77e9b1c7df5b00fb02043f3985726b7281b2fb09",
    "9d23bed233cba5f4bc77671e9a8f738919f994b42e7c17138d6a9c680529edc1",
    "0c16d6213b8f578cf77373e913bbc63ad566366b325e532eb74460718768aaf3",
    "d70e89d314560717b8a851f654af4f2a0c8d015766b6548c661c35668f5c0e63",
    "a47b0573c46bc9463ed187369bfe43dd6d683cce83a9e59fa0a07a51700d17af",
))))


@dataclass(frozen=True)
class MembershipRow:
    image_id: str
    identity_id: str
    component_id: str
    design_id: str
    label: int


@dataclass(frozen=True)
class ProtocolAContract:
    fold: int
    ref_group: str
    records_sha256: str
    split_sha256: str
    source_designs: tuple[str, ...]
    source_train: tuple[MembershipRow, ...]
    source_val: tuple[MembershipRow, ...]
    target_test: tuple[MembershipRow, ...]
    eligible_support: tuple[MembershipRow, ...]
    forbidden_components: tuple[str, ...]


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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _authenticated(path, expected, name):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{name} must be regular metadata")
    _require(not any(parent.is_symlink() for parent in path.parents), f"{name} symlink path")
    content = path.read_bytes()
    _require(hashlib.sha256(content).hexdigest() == expected, f"{name} hash mismatch")
    return content


def load_protocol_a_contract(records_csv, split_csv, *, fold, records_sha256):
    """Authenticate metadata, preserving published related classifier components."""
    _require(type(fold) is int and fold in REVIEWED_SPLIT_SHA256, "fold must be an integer in 0..9")
    _require(records_sha256 == RECORDS_SHA256, "records must match reviewed original authority")
    split = pd.read_csv(BytesIO(_authenticated(split_csv, REVIEWED_SPLIT_SHA256[fold], "split")),
                        dtype=str, keep_default_na=False)
    records = pd.read_csv(BytesIO(_authenticated(records_csv, records_sha256, "records")),
                          dtype=str, keep_default_na=False)
    _require({"image_id", "identity_id", "component_id", "design_id", "label", "dataset", "source_id", "donor_id"} <= set(records)
             and {"image_id", "component_id", "design_id", "label", "protocol", "role", "ref_group"} <= set(split),
             "missing membership/provenance columns")
    for frame in (records, split):
        _require(len(frame) > 0 and frame.image_id.is_unique, "duplicate or empty membership")
        for field in ("image_id", "component_id", "design_id"):
            _require(frame[field].str.strip().ne("").all(), f"empty {field}")
        _require(frame.label.isin(("0", "1")).all(), "binary labels required")
    _require(records.identity_id.str.strip().ne("").all(), "empty identity_id")
    for field in ("source_id", "donor_id"):
        _require(set(records[field]) - {""} <= set(records.identity_id), f"unknown {field} identity")
    components = provenance_components(records)
    _require((components.groupby(records.component_id).nunique() == 1).all()
             and (records.component_id.groupby(components).nunique() == 1).all(), "invalid transitive provenance components")
    _require(split.protocol.eq("A").all() and split.ref_group.eq(f"it{fold}").all()
             and set(split.role) == {"train", "val", "test"}, "official A fold/roles required")
    if "fold" in split:
        _require(split["fold"].eq(str(fold)).all(), "explicit split fold mismatch")
    sidtd = records.loc[records.dataset == "sidtd"].set_index("image_id")
    _require(set(sidtd.index) == set(split.image_id), "records/split membership mismatch")
    joined = split.join(sidtd[["identity_id", "component_id", "design_id", "label"]],
                        on="image_id", rsuffix="_record", validate="one_to_one")
    for field in ("component_id", "design_id", "label"):
        _require(joined[field].eq(joined[field + "_record"]).all(), f"records/split {field} mismatch")
    members = {role: tuple(MembershipRow(row.image_id, row.identity_id, row.component_id, row.design_id, int(row.label))
                           for row in joined.loc[joined.role == role].sort_values("image_id").itertuples())
               for role in ("train", "val", "test")}
    forbidden = tuple(sorted({row.component_id for row in members["val"] + members["test"]}))
    eligible = tuple(row for row in members["train"] if row.label == 0 and row.component_id not in forbidden)
    designs = tuple(sorted(set(joined.design_id)))
    _require(set(row.design_id for row in eligible) == set(designs), "empty genuine training support design")
    return ProtocolAContract(fold, f"it{fold}", records_sha256, REVIEWED_SPLIT_SHA256[fold], designs,
                             members["train"], members["val"], members["test"], eligible, forbidden)


def training_accounting(contract, recipe):
    _require(isinstance(contract, ProtocolAContract), "typed A contract required")
    fixed = dict(epochs=100, batch_size=8, eval_batch_size=1, accumulation_steps=1,
                 drop_last=True, eval_drop_last=False, seed=42)
    _require(isinstance(recipe, dict) and all(type(recipe.get(key)) is type(value) and recipe[key] == value
             for key, value in fixed.items()), "unsupported Protocol A recipe")
    scheduler = recipe.get("scheduler")
    _require(isinstance(scheduler, dict) and scheduler.get("name") == "warmup_cosine"
             and type(scheduler.get("warmup_iters")) is int and scheduler["warmup_iters"] == 1000
             and type(scheduler.get("warmup_cap_fraction")) in (int, float)
             and scheduler["warmup_cap_fraction"] == 0.1, "unsupported scheduler recipe")
    batches = len(contract.source_train) // 8
    _require(batches > 0 and len(contract.source_val) > 0, "empty training/validation population")
    total = batches * 100
    return TrainingAccounting(len(contract.source_train), len(contract.source_val), len(contract.target_test),
        batches, batches * 8, len(contract.source_train) - batches * 8, total, min(1000, int(0.1 * total)))


def make_protocol_a_scheduler(optimizer, contract, recipe):
    from .pilots import pilot_scheduler

    return pilot_scheduler(optimizer, "D_running_pilot", training_accounting(contract, recipe).total_optimizer_steps)
