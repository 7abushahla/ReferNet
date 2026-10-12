"""Fold-native A data consumer using unchanged enrollment and query geometry.

Metadata authority is not training permission. The checked runner owns phase and
GPU gates. Optional protocol_a_evaluation_v1 authority is issued only after both
100-epoch parents and fresh replays have been verified by that runner. This
consumer authenticates checkpoint bytes, not their completion or selection.

Historical bank manifests remain byte-for-byte unchanged. Explicit, hash-bound
metadata and product-root relocation documents authorize durable locations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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

from ..protocol_a_contract import ProtocolAContract, load_protocol_a_contract
from .direct_reference import (
    _require, _bytes, _json, _digest, _path, _image, _common, _overlap, _available,
    _fingerprint, _augmentation, _isolated_rng, _policy as original_policy,
)
from . import enrollment
from .enrollment import _fit, _similarity, _warp, resolve_enrollment_bank
from .references import fold_of
from .rectify import to_canvas
from .register import _features, MIN_INLIERS
from .dataset import PairDataset, _normalize, _photometric
from .immutable_array_cache import ImmutableArrayCache

SCHEMA = "protocol_a_registration_v1"
PRODUCTS = {"target", "target_coverage", "target_valid", "coverage", "valid", "zone", "median", "geometry", "manifest"}


@dataclass(frozen=True)
class ReferenceContext:
    contract: ProtocolAContract
    raw_root: Path
    _manifest: str
    _rows: str
    _bindings: str
    identities: str
    product_root: str
    evaluation_authorized: bool

    @property
    def manifest(self):
        return json.loads(self._manifest)

    @property
    def bindings(self):
        return json.loads(self._bindings)

    def rows(self, role):
        _require(role in ("train", "val", "test"), "invalid Protocol A role")
        _require(role != "test" or self.evaluation_authorized, "test pixels not authorized")
        return pd.DataFrame(json.loads(self._rows)[role])


def _policy():
    value = original_policy()
    value.update(version="protocol_a_query_canvas_v1", protocol="A",
                 implementation_sha256=_digest(dict(consumer=Path(__file__).read_text(),
                                                     original=value["implementation_sha256"])))
    return value


def _typed_equal(actual, expected):
    return type(actual) is type(expected) and actual == expected


def _historical_path(value):
    _require(isinstance(value, str) and value and "\\" not in value,
             "historical absolute canonical path required")
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts and str(path) == value,
             "historical absolute canonical path required")
    return path


def _assert_context(context):
    _require(isinstance(context, ReferenceContext) and isinstance(context.contract, ProtocolAContract),
             "typed authenticated A context required")
    for binding in context.bindings:
        _bytes(binding)


def _metadata_paths(manifest, enrollment_binding, paths, identities, relocation_binding):
    supplied = manifest.get("input_files", {})
    _require({"records", "split"} <= set(supplied), "original preparation metadata bindings required")
    expected = {}
    for name, actual in paths.items():
        original = supplied[name]
        _require(original.get("sha256") == actual["sha256"], "prepared metadata hash mismatch")
        original_path = _historical_path(original.get("path"))
        actual_path = _path(actual["path"])
        if original_path != actual_path:
            expected[name] = dict(original_path=str(original_path), **actual)
    if relocation_binding is None:
        _require(not expected, "explicit metadata relocation required")
    else:
        relocation = _json(relocation_binding)
        wanted = dict(schema_version="protocol_a_metadata_relocation_v1", protocol="A",
            fold=identities["fold"], enrollment_manifest_sha256=enrollment_binding["sha256"], relocations=expected)
        _require(relocation == wanted and type(relocation.get("fold")) is int,
                 "metadata relocation differs from original/actual authorities")


def _enrollment_metadata(manifest, contract, records, split):
    expected = dict(schema_version=enrollment.SCHEMA_VERSION, geometry_version=enrollment.GEOMETRY_VERSION,
        fold=contract.fold, official_fold_binding=dict(protocol="A", fold=contract.fold, ref_group=contract.ref_group),
        official_split_sha256=enrollment._table_hash(split), official_split_hash_format="canonical_json_rows_v1",
        provenance_graph_sha256=enrollment._table_hash(records[["image_id", "identity_id", "source_id", "donor_id", "component_id"]]),
        classifier_counts=dict(train=len(contract.source_train), val=len(contract.source_val), test=len(contract.target_test)),
        forbidden_components=list(contract.forbidden_components),
        eligible_ids=[row.image_id for row in contract.eligible_support])
    for name, value in expected.items():
        _require(_typed_equal(manifest.get(name), value), f"enrollment {name} differs from official A metadata")
    _require(type(manifest["official_fold_binding"].get("fold")) is int, "fold binding must be integer")
    index = records.set_index("image_id")
    contributors = manifest.get("contributors", {})
    _require(set(contributors) == set(expected["eligible_ids"]), "foreign or missing support contributor")
    for member in contract.eligible_support:
        row = index.loc[member.image_id]
        contributor = contributors[member.image_id]
        for field in ("image_id", "dataset", "design_id", "identity_id", "component_id", "path", "annotation", "width", "height"):
            wanted = member.image_id if field == "image_id" else row[field]
            _require(contributor.get(field) == wanted, "contributor metadata mismatch")
        for field in ("raw_sha256", "layout_sha256"):
            _require(isinstance(contributor.get(field), str) and len(contributor[field]) == 64
                     and all(character in "0123456789abcdef" for character in contributor[field]), "invalid contributor digest")
    banks = manifest.get("banks", {})
    _require(set(banks) == set(contract.source_designs), "unexpected support designs")
    for design, named in banks.items():
        _require(set(named) == set(enrollment.BANKS), "three A cross-fit/full banks required")
        support = [row for row in contract.eligible_support if row.design_id == design]
        for name, bank in named.items():
            members = support if name == "enrollment" else [row for row in support if fold_of(row.component_id) == int(name[-1])]
            identifiers = [row.image_id for row in members]
            wanted = dict(bank=name, design_id=design, fold=contract.fold,
                geometry_version=enrollment.GEOMETRY_VERSION, support_ids=identifiers,
                component_ids=sorted({row.component_id for row in members}),
                contributors={identifier: contributors[identifier] for identifier in identifiers})
            _require(identifiers and all(_typed_equal(bank.get(key), value) for key, value in wanted.items()),
                     "bank support/component/fold mismatch")
            registered = bank.get("registered_ids")
            _require(isinstance(registered, list) and registered and registered == sorted(set(registered))
                     and set(registered) <= set(identifiers), "invalid registered support membership")
            counts = dict(authorized=len(identifiers), registered=len(registered), failed=len(identifiers) - len(registered))
            _require(bank.get("counts") == counts and all(type(value) is int for value in bank["counts"].values()),
                     "bank counts mismatch")
    source = manifest.get("source", {})
    _require(source.get("implementation_sha256") == hashlib.sha256(Path(enrollment.__file__).read_bytes()).hexdigest()
             and source.get("helpers") == {name: hashlib.sha256(Path(enrollment.__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ("records.py", "rectify.py", "references.py", "register.py")},
             "bank scientific helper source mismatch")


def _products(bank, context):
    result = copy.deepcopy(bank.get("products", {}))
    _require(set(result) == PRODUCTS, "complete nine median bank products required")
    original = _historical_path(context.manifest["output_root"])
    expected = original / f"fold{context.contract.fold}" / bank["design_id"] / bank["bank"]
    relocated = _path(context.product_root)
    for name, binding in result.items():
        suffix = ".json" if name in ("geometry", "manifest") else ".png"
        path = _historical_path(binding.get("path"))
        _require(path == expected / (name + suffix), "product crosses authorized bank/root")
        binding["path"] = str(relocated / path.relative_to(original))
        _bytes(binding)
    geometry = _json(result["geometry"])
    expected_fields = dict(schema_version=enrollment.SCHEMA_VERSION, geometry_version=enrollment.GEOMETRY_VERSION,
        fold=context.contract.fold, design_id=bank["design_id"], bank=bank["bank"], canvas_hw=[512, 800],
        axes="x=column,y=row; native_to_bank=canvas_to_bank@native_to_canvas",
        valid_interpolation="nearest", rgb_interpolation="linear", support_ids=bank["support_ids"],
        registered_ids=bank["registered_ids"], contributors=bank["contributors"],
        source=context.manifest["source"], official_split_sha256=context.manifest["official_split_sha256"],
        anchor_id=bank["support_ids"][0])
    _require(all(_typed_equal(geometry.get(key), value) for key, value in expected_fields.items()), "bank geometry identity mismatch")
    registrations = geometry.get("registrations", {})
    _require(set(registrations) == set(bank["support_ids"]), "geometry support registrations mismatch")
    for identifier, registration in registrations.items():
        wanted = identifier in bank["registered_ids"]
        _require(type(registration.get("target_applied")) is bool and registration["target_applied"] == wanted,
                 "geometry registered membership mismatch")
        for field in ("canvas_to_bank", "native_to_bank"):
            matrix = registration.get(field)
            if wanted:
                _similarity(np.asarray(matrix, dtype=np.float64))
            else:
                _require(matrix is None, "failed support has transform")
    dependency = _json(result["manifest"])
    _require(dependency == dict(geometry, output_sha256={name: binding["sha256"] for name, binding in result.items()
                                                       if name != "manifest"}), "bank dependency/geometry hashes mismatch")
    return result


def load_reference_context(records_csv, split_csv, *, fold, records_sha256, enrollment_binding,
                           raw_root, root_binding=None, metadata_relocation_binding=None, evaluation_binding=None):
    contract = load_protocol_a_contract(records_csv, split_csv, fold=fold, records_sha256=records_sha256)
    paths = dict(records=dict(path=str(_path(str(Path(records_csv).absolute()))), sha256=records_sha256),
                 split=dict(path=str(_path(str(Path(split_csv).absolute()))), sha256=contract.split_sha256))
    records = pd.read_csv(BytesIO(_bytes(paths["records"])), keep_default_na=False)
    split = pd.read_csv(BytesIO(_bytes(paths["split"])), keep_default_na=False)
    manifest = _json(enrollment_binding)
    _enrollment_metadata(manifest, contract, records, split)
    identities = dict(protocol="A", fold=fold, records_sha256=records_sha256,
                      split_sha256=contract.split_sha256, enrollment_manifest_sha256=enrollment_binding["sha256"])
    _metadata_paths(manifest, enrollment_binding, paths, identities, metadata_relocation_binding)
    original = _historical_path(manifest["output_root"])
    relocated = original
    bindings = [*paths.values(), copy.deepcopy(enrollment_binding)]
    if metadata_relocation_binding is not None:
        bindings.append(copy.deepcopy(metadata_relocation_binding))
    if root_binding is not None:
        root = _json(root_binding)
        _require(root.get("schema_version") == "protocol_a_product_root_v1"
                 and root.get("protocol") == "A" and type(root.get("fold")) is int and root["fold"] == fold
                 and root.get("enrollment_manifest_sha256") == enrollment_binding["sha256"]
                 and _historical_path(root.get("original_root")) == original, "invalid A product relocation")
        relocated = _path(root.get("relocated_root"))
        bindings.append(copy.deepcopy(root_binding))
    rows, index = {}, records.set_index("image_id")
    for role, members in (("train", contract.source_train), ("val", contract.source_val), ("test", contract.target_test)):
        rows[role] = []
        for member in members:
            row = index.loc[member.image_id].to_dict()
            row.update(image_id=member.image_id, role=role, label=member.label)
            _require({"path", "attack_family"} <= set(row), "query path/family metadata required")
            relative = Path(row["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in row["path"]
                     and relative.parts and relative.parts[0] == "sidtd", "unsafe raw query path")
            if role != "train":
                for field in ("annotation", "component_id", "identity_id", "source_id", "donor_id"):
                    row.pop(field, None)
            rows[role].append(row)
    if evaluation_binding is not None:
        evaluation = _json(evaluation_binding)
        _require(evaluation.get("schema_version") == "protocol_a_evaluation_v1"
                 and all(_typed_equal(evaluation.get(key), value) for key, value in identities.items())
                 and "group" not in evaluation and evaluation.get("source_validation_selected") is True
                 and evaluation.get("authorized_image_ids") == [row.image_id for row in contract.target_test],
                 "invalid locked A test authorization")
        checkpoints = evaluation.get("checkpoints", {})
        _require(set(checkpoints) == {"off", "on"}, "both selected checkpoints required")
        for binding in checkpoints.values():
            _bytes(binding)
        bindings.extend([copy.deepcopy(evaluation_binding), *copy.deepcopy(list(checkpoints.values()))])
    context = ReferenceContext(contract, _path(str(Path(raw_root).absolute())), json.dumps(manifest, sort_keys=True),
        json.dumps(rows, sort_keys=True), json.dumps(bindings, sort_keys=True), json.dumps(identities, sort_keys=True),
        str(relocated), evaluation_binding is not None)
    for named in manifest["banks"].values():
        for bank in named.values():
            _products(bank, context)
    return context


def _roles(context, roles):
    _require(isinstance(roles, (tuple, list)) and roles and len(set(roles)) == len(roles), "explicit unique roles required")
    _require(set(roles) <= {"train", "val", "test"} and not ("test" in roles and len(roles) != 1),
             "test registration must be isolated from training")
    _require("test" not in roles or context.evaluation_authorized, "test pixels not authorized")
    return [row for role in roles for row in context.rows(role).to_dict("records")]


def _binding(context, row, query_sha, bank, products):
    value = dict(image_id=row["image_id"], path=row["path"], role=row["role"], design_id=row["design_id"],
                 fold=context.contract.fold, bank=bank["bank"], query_sha256=query_sha, products=products,
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


def build_protocol_a_registration(context, *, roles, output_path):
    _assert_context(context)
    rows = _roles(context, roles)
    _require(not Path(output_path).exists(), "registration output already exists")
    entries = {}
    manifest, banks = context.manifest, {}
    for row in rows:
        bank = resolve_enrollment_bank(manifest, row["design_id"], row["role"],
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
    return load_protocol_a_registration(context, binding, roles=roles)


def load_protocol_a_registration(context, binding, *, roles):
    _assert_context(context)
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
        bank = resolve_enrollment_bank(manifest, row["design_id"], row["role"],
                                      row.get("component_id") if row["role"] == "train" else None)
        bank_key = (row["design_id"], bank["bank"])
        if bank_key not in banks:
            banks[bank_key] = _products(bank, context)
        products = banks[bank_key]
        expected = _binding(context, row, entry["query_sha256"], bank, products)
        _require(set(entry) == set(expected) | {"key", "status", "reason", "inliers", "query_canvas_to_bank"}
                 and all(entry.get(key) == value for key, value in expected.items())
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


class ProtocolAReferenceDataset(Dataset):
    def __init__(self, context, table, *, role, reference_enabled, augment, cache_bytes=0):
        _require(isinstance(context, ReferenceContext) and isinstance(table, RegistrationTable), "authenticated context/table required")
        _require(type(reference_enabled) is bool, "reference_enabled must be boolean")
        _require(role != "test" or context.evaluation_authorized, "test pixels not authorized")
        _assert_context(context)
        rows = context.rows(role).to_dict("records")
        document = json.loads(table.serialized)
        _require(table.binding_serialized is not None, "authenticated table binding required")
        validated = load_protocol_a_registration(context, json.loads(table.binding_serialized), roles=document.get("roles"))
        _require(validated.serialized == table.serialized, "dataset table metadata mismatch")
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
        _require(role == "train" or not self.augment["enabled"], "evaluation augmentation forbidden")
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
                _require(observed == _fingerprint(binding), "authority changed during authentication")
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
                before = _fingerprint(binding)
                _bytes(binding)
                _require(before == _fingerprint(binding), "input changed during authentication")

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
                                       identities=json.loads(self.context.identities)), validate=validate, build=build)
        validate()
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
                          attack_family=row["attack_family"], fold=self.context.contract.fold, bank=entry["bank"],
                          registration_key=entry["key"], registration_table_sha256=self.table_sha256,
                          query_sha256=entry["query_sha256"], availability_reason=reason, requires_referral=not available))
        if self.reference_enabled:
            result["reference"] = torch.from_numpy(np.ascontiguousarray(_normalize(reference))) if available else torch.zeros((3, 512, 800), dtype=torch.float32)
        return result

