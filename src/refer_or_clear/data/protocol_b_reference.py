"""Protocol B query-canvas consumer with explicit enrollment and role authority."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..protocol_b_contract import load_protocol_b_contract, load_target_enrollment
from .protocol_b_enrollment import make_protocol_b_enrollment_manifest, resolve_protocol_b_bank
from .direct_reference import (
    _require, _bytes, _json, _digest, _path, _image, _common, _overlap, _available,
    _fingerprint, _augmentation, _isolated_rng, _policy as original_policy,
)
from .enrollment import _fit, _similarity, _warp
from .rectify import to_canvas
from .register import _features, MIN_INLIERS
from .dataset import PairDataset, _normalize, _photometric
from .immutable_array_cache import ImmutableArrayCache

SCHEMA = "protocol_b_registration_v1"


@dataclass(frozen=True)
class ReferenceContext:
    contract: object
    target: object
    raw_root: Path
    _manifest: str
    _rows: str
    _bindings: str
    identities: str
    evaluation_authorized: bool

    @property
    def manifest(self):
        return json.loads(self._manifest)

    @property
    def bindings(self):
        return json.loads(self._bindings)

    def rows(self, role):
        _require(role in ("train", "val", "test"), "invalid Protocol B role")
        _require(role != "test" or self.evaluation_authorized, "test pixels not authorized")
        return pd.DataFrame(json.loads(self._rows)[role])


@lru_cache(maxsize=1)
def _policy():
    value = original_policy()
    value.update(version="protocol_b_query_canvas_v1", protocol="B",
                 implementation_sha256=_digest(dict(consumer=Path(__file__).read_text(),
                                                     original=value["implementation_sha256"])))
    return value


def _products(bank, context):
    result = copy.deepcopy(bank.get("products", {}))
    _require(set(result) == {"target", "target_coverage", "target_valid", "coverage", "valid", "zone", "median", "geometry", "manifest"},
             "complete median bank products required")
    root = _path(context.manifest["output_root"])
    expected_root = root / f"group{context.contract.group}" / bank["design_id"] / bank["bank"]
    for name, binding in result.items():
        suffix = ".json" if name in ("geometry", "manifest") else ".png"
        _require(_path(binding["path"]) == expected_root / (name + suffix), "product crosses authorized bank/root")
        _bytes(binding)
    geometry = _json(result["geometry"])
    _require(geometry.get("schema_version") == "protocol_b_enrollment_v1"
             and geometry.get("protocol") == "B" and type(geometry.get("group")) is int
             and geometry["group"] == context.contract.group and "fold" not in geometry
             and geometry.get("design_id") == bank["design_id"] and geometry.get("bank") == bank["bank"]
             and geometry.get("support_ids") == bank["support_ids"]
             and geometry.get("contributors") == bank.get("contributors"), "bank geometry identity mismatch")
    dependency = _json(result["manifest"])
    _require(dependency.get("output_sha256") == {name: binding["sha256"] for name, binding in result.items() if name != "manifest"},
             "bank dependency hashes differ from products")
    return result


def load_reference_context(records_csv, split_csv, *, group, records_sha256,
                           target_binding, enrollment_binding, raw_root, evaluation_binding=None):
    contract = load_protocol_b_contract(Path(records_csv), Path(split_csv), group=group,
                                       records_sha256=records_sha256)
    target = load_target_enrollment(contract, Path(target_binding["path"]),
                                    expected_sha256=target_binding["sha256"])
    manifest = _json(enrollment_binding)
    expected = make_protocol_b_enrollment_manifest(Path(records_csv), contract, target, Path(raw_root))
    for name, value in expected.items():
        if name not in ("banks", "contributors"):
            _require(manifest.get(name) == value, f"enrollment {name} differs from authorized metadata")
    _require(set(manifest.get("banks", {})) == set(expected["banks"]), "unexpected enrollment designs")
    _require(set(manifest.get("contributors", {})) == set(expected["contributors"]), "foreign contributor")
    for identifier, contributor in expected["contributors"].items():
        observed = manifest["contributors"][identifier]
        _require(all(observed.get(key) == value for key, value in contributor.items()), "contributor metadata mismatch")
    root = _path(str(raw_root))
    records = pd.read_csv(BytesIO(_bytes(dict(path=str(Path(records_csv).resolve()), sha256=records_sha256))),
                          keep_default_na=False)
    index = records.set_index("image_id")
    rows = {}
    for role, members in (("train", contract.source_train), ("val", contract.source_val),
                          ("test", tuple(row for row in contract.target_test if row.image_id in target.query_ids))):
        rows[role] = []
        for member in members:
            row = index.loc[member.image_id].to_dict()
            row.update(image_id=member.image_id, role=role, label=member.label)
            if role != "train":
                for field in ("annotation", "component_id", "identity_id", "source_id", "donor_id"):
                    row.pop(field, None)
            relative = Path(row["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in row["path"],
                     "unsafe raw query path")
            rows[role].append(row)
    identities = dict(protocol="B", group=group, records_sha256=records_sha256,
                      split_sha256=contract.split_sha256, target_enrollment_sha256=target.manifest_sha256,
                      enrollment_manifest_sha256=enrollment_binding["sha256"])
    bindings = [dict(path=str(Path(records_csv).resolve()), sha256=records_sha256),
                dict(path=str(Path(split_csv).resolve()), sha256=contract.split_sha256),
                target_binding, enrollment_binding]
    if evaluation_binding is not None:
        evaluation = _json(evaluation_binding)
        _require(evaluation.get("schema_version") == "protocol_b_evaluation_v1"
                 and all(type(evaluation.get(key)) is type(value) and evaluation[key] == value for key, value in identities.items())
                 and evaluation.get("source_validation_selected") is True
                 and evaluation.get("authorized_image_ids") == sorted(target.query_ids),
                 "invalid locked Protocol B evaluation authorization")
        checkpoints = evaluation.get("checkpoints", {})
        _require(set(checkpoints) == {"off", "on"}, "both selected checkpoints required")
        for binding in checkpoints.values():
            _bytes(binding)
        bindings.extend([evaluation_binding, *checkpoints.values()])
    context = ReferenceContext(contract, target, root, json.dumps(manifest, sort_keys=True),
                               json.dumps(rows, sort_keys=True), json.dumps(bindings, sort_keys=True),
                               json.dumps(identities, sort_keys=True), evaluation_binding is not None)
    for design, banks in manifest["banks"].items():
        _require(set(banks) == set(expected["banks"][design]), "unexpected enrollment banks")
        for name, bank in banks.items():
            if name.startswith("train_half"):
                other = next(row.component_id for row in contract.source_train
                             if row.design_id == design and row.component_id not in bank["component_ids"])
                selected = resolve_protocol_b_bank(manifest, design, "train", other)
            else:
                selected = resolve_protocol_b_bank(manifest, design,
                                                    "val" if design in contract.source_designs else "test")
            _require(selected["bank"] == name, "cross-fit bank mismatch")
            _products(selected, context)
    return context


def _roles(context, roles):
    _require(isinstance(roles, (tuple, list)) and roles and len(set(roles)) == len(roles), "explicit unique roles required")
    _require(set(roles) <= {"train", "val", "test"} and not ("test" in roles and len(roles) != 1),
             "test registration must be isolated from training")
    _require("test" not in roles or context.evaluation_authorized, "test pixels not authorized")
    return [row for role in roles for row in context.rows(role).to_dict("records")]


def _binding(context, row, query_sha, bank, products):
    value = dict(image_id=row["image_id"], path=row["path"], role=row["role"], design_id=row["design_id"],
                 group=context.contract.group, bank=bank["bank"], query_sha256=query_sha, products=products,
                 identities=json.loads(context.identities), policy=_policy())
    if row["role"] == "train":
        value["training_component"] = row["component_id"]
    return value


@dataclass(frozen=True)
class RegistrationTable(Mapping):
    serialized: str
    authority_sha256: str
    binding_serialized: str | None = None

    def __getitem__(self, key):
        return json.loads(self.serialized)[key]

    def __iter__(self):
        return iter(json.loads(self.serialized))

    def __len__(self):
        return len(json.loads(self.serialized))


def build_protocol_b_registration(context, *, roles, output_path):
    rows = _roles(context, roles)
    _require(not Path(output_path).exists(), "registration output already exists")
    entries = {}
    manifest, banks = context.manifest, {}
    for row in rows:
        bank = resolve_protocol_b_bank(manifest, row["design_id"], row["role"],
                                      row.get("component_id") if row["role"] == "train" else None)
        bank_key = (row["design_id"], bank["bank"])
        if bank_key not in banks:
            products = _products(bank, context)
            target, masks = _common(products)
            banks[bank_key] = (products, _features(target), masks)
        products, target_features, masks = banks[bank_key]
        path = _path(str(context.raw_root / row["path"]))
        query_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        query, valid, unused_transform = to_canvas(_image(dict(path=str(path), sha256=query_sha), cv2.IMREAD_COLOR))
        binding = _binding(context, row, query_sha, bank, products)
        key = _digest(binding)
        inliers = 0
        try:
            matrix, inliers = _fit(_features(query), target_features, int(key[:8], 16) & 0x7fffffff)
            _require(matrix is not None and type(inliers) is int and inliers >= MIN_INLIERS, "registration failed")
            matrix = _similarity(matrix)
            unused_inverse, overlap = _overlap(valid, masks, matrix)
            reason = "fit" if _available(overlap) else "empty_overlap"
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            matrix, reason = None, "registration_failed"
            inliers = inliers if type(inliers) is int and inliers >= 0 else 0
        entries[row["image_id"]] = dict(binding, key=key, status="fit" if reason == "fit" else "failed",
                                        reason=reason, inliers=inliers,
                                        query_canvas_to_bank=matrix.tolist() if reason == "fit" else None)
    document = dict(schema_version=SCHEMA, roles=list(roles), identities=json.loads(context.identities),
                    policy=_policy(), entries=entries)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    binding = dict(path=str(Path(output_path).resolve()),
                   sha256=hashlib.sha256(Path(output_path).read_bytes()).hexdigest())
    return load_protocol_b_registration(context, binding, roles=roles)


def load_protocol_b_registration(context, binding, *, roles):
    rows = _roles(context, roles)
    table = _json(binding)
    _require(table.get("schema_version") == SCHEMA and table.get("roles") == list(roles)
             and table.get("identities") == json.loads(context.identities) and table.get("policy") == _policy(),
             "registration schema/roles/source mismatch")
    entries = table.get("entries", {})
    _require(set(entries) == {row["image_id"] for row in rows}, "registration query membership mismatch")
    manifest, banks = context.manifest, {}
    for row in rows:
        entry = entries[row["image_id"]]
        bank = resolve_protocol_b_bank(manifest, row["design_id"], row["role"],
                                      row.get("component_id") if row["role"] == "train" else None)
        bank_key = (row["design_id"], bank["bank"])
        if bank_key not in banks:
            banks[bank_key] = _products(bank, context)
        products = banks[bank_key]
        expected = _binding(context, row, entry["query_sha256"], bank, products)
        _require(all(entry.get(key) == value for key, value in expected.items())
                 and entry.get("key") == _digest(expected), "registration query/bank binding mismatch")
        _bytes(dict(path=str(context.raw_root / row["path"]), sha256=entry["query_sha256"]))
        _require(type(entry.get("inliers")) is int and entry["inliers"] >= 0, "invalid registration inliers")
        if entry.get("status") == "fit":
            _require(entry.get("reason") == "fit" and entry["inliers"] >= MIN_INLIERS, "invalid successful fit")
            _similarity(np.asarray(entry["query_canvas_to_bank"], np.float64))
        else:
            _require(entry.get("status") == "failed" and entry.get("reason") in ("registration_failed", "empty_overlap")
                     and entry.get("query_canvas_to_bank") is None, "invalid registration failure")
        _require(row["role"] == "train" or "training_component" not in entry, "evaluation provenance routing forbidden")
    return RegistrationTable(json.dumps(table, sort_keys=True), binding["sha256"], json.dumps(binding, sort_keys=True))


class ProtocolBReferenceDataset(Dataset):
    def __init__(self, context, table, *, role, reference_enabled, augment, cache_bytes=0):
        _require(isinstance(context, ReferenceContext) and isinstance(table, RegistrationTable), "authenticated context/table required")
        _require(type(reference_enabled) is bool, "reference_enabled must be boolean")
        _require(role != "test" or context.evaluation_authorized, "test pixels not authorized")
        rows = context.rows(role).to_dict("records")
        document = json.loads(table.serialized)
        _require(table.binding_serialized is not None
                 and json.loads(table.binding_serialized)["sha256"] == table.authority_sha256
                 and _json(json.loads(table.binding_serialized)) == document
                 and document.get("schema_version") == SCHEMA and document.get("policy") == _policy(),
                 "dataset table differs from authenticated bytes")
        _require(role in document["roles"] and document["identities"] == json.loads(context.identities), "foreign dataset table")
        self.context, self.role, self.reference_enabled = context, role, reference_enabled
        self._rows = tuple(json.dumps(row, sort_keys=True) for row in rows)
        self._entries = tuple(json.dumps(document["entries"][row["image_id"]], sort_keys=True) for row in rows)
        self.table_sha256 = table.authority_sha256
        self.authority_files = context.bindings + [json.loads(table.binding_serialized)]
        self.authority_fingerprints = []
        for binding in self.authority_files:
            _bytes(binding)
            self.authority_fingerprints.append(_fingerprint(binding))
        self.cache = ImmutableArrayCache(max_bytes=cache_bytes)
        supplied = dict(augment)
        seed = supplied.pop("seed", 42)
        _require(type(seed) is int, "integer augmentation seed required")
        self.augment = _augmentation(supplied)
        self.train = role == "train"
        self.crop, self.affine_px = None, self.augment["affine_px"]
        self.geometry_rng, self.photometry_rng = random.Random(seed), random.Random(seed + 1)

    def __len__(self):
        return len(self._rows)

    @property
    def rows(self):
        return [json.loads(row) for row in self._rows]

    def __getitem__(self, index):
        for position, binding in enumerate(self.authority_files):
            observed = _fingerprint(binding)
            if observed != self.authority_fingerprints[position]:
                _bytes(binding)
                self.authority_fingerprints[position] = observed
        row = json.loads(self._rows[index])
        entry = json.loads(self._entries[index])
        products = entry["products"]
        query_binding = dict(path=str(self.context.raw_root / row["path"]), sha256=entry["query_sha256"])
        used = [query_binding, *(products[name] for name in ("geometry", "manifest", "target_valid", "valid", "zone"))]
        if self.reference_enabled and entry["status"] == "fit":
            used.append(products["median"])

        def validate():
            for binding in used:
                _bytes(binding)

        def build():
            unused_target, masks = _common(products, registration=False)
            query, valid, unused_transform = to_canvas(_image(query_binding, cv2.IMREAD_COLOR))
            overlap, zone = np.zeros((512, 800), np.uint8), np.zeros((512, 800), np.uint8)
            arrays = dict(query=query, overlap=overlap, zone=zone)
            if entry["status"] == "fit":
                inverse, overlap = _overlap(valid, masks, np.asarray(entry["query_canvas_to_bank"]))
                arrays.update(overlap=overlap, zone=(_warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap)
                if self.reference_enabled and _available(overlap):
                    median = _image(products["median"], cv2.IMREAD_COLOR)
                    _require(median.shape == (512, 800, 3), "median must be 512x800 BGR")
                    arrays["reference"] = _warp(median, inverse)
            return arrays

        arrays = self.cache.fetch(dict(entry=entry, reference_enabled=self.reference_enabled,
                                       identities=json.loads(self.context.identities)), validate=validate, build=build,
                                   validate_hits=False)
        query, zone, overlap = arrays["query"], arrays["zone"], arrays["overlap"]
        reference = arrays.get("reference")
        if self.train and self.augment["enabled"]:
            with _isolated_rng(self.geometry_rng):
                matrix, window = PairDataset._geometry(self, 512, 800)
                query = PairDataset._warp(query, matrix, window, False)
                zone = PairDataset._warp(zone, matrix, window, True)
                overlap = PairDataset._warp(overlap, matrix, window, True)
                if reference is not None:
                    reference = PairDataset._warp(reference, matrix, window, False)
                if self.augment["jitter_px"] and random.random() < self.augment["jitter_p"]:
                    jitter = np.float32([[1, 0, random.randint(-self.augment["jitter_px"], self.augment["jitter_px"])],
                                         [0, 1, random.randint(-self.augment["jitter_px"], self.augment["jitter_px"])]])
                    query = cv2.warpAffine(query, jitter, (800, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    if reference is not None:
                        reference = cv2.warpAffine(reference, jitter, (800, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    zone = cv2.warpAffine(zone, jitter, (800, 512), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    overlap = cv2.warpAffine(overlap, jitter, (800, 512), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if self.augment["photometric"]:
                with _isolated_rng(self.photometry_rng):
                    query = _photometric(query, .2)
        available = _available(overlap)
        reason = entry["reason"]
        if not available:
            zone.fill(0)
            overlap.fill(0)
            if reason == "fit":
                reason = "empty_overlap"
        zone *= overlap
        result = dict(query=torch.from_numpy(np.ascontiguousarray(np.concatenate([_normalize(query), zone[None].astype(np.float32)]))),
                      overlap=torch.from_numpy(overlap[None].astype(np.float32)), context_available=torch.tensor(available, dtype=torch.bool),
                      label=int(row["label"]), meta=dict(image_id=row["image_id"], design_id=row["design_id"], role=self.role,
                          attack_family=row["attack_family"], group=self.context.contract.group, bank=entry["bank"],
                          registration_key=entry["key"], registration_table_sha256=self.table_sha256,
                          query_sha256=entry["query_sha256"], availability_reason=reason, requires_referral=not available))
        if self.reference_enabled:
            result["reference"] = torch.from_numpy(np.ascontiguousarray(_normalize(reference))) if available else torch.zeros((3, 512, 800), dtype=torch.float32)
        return result
