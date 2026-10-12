"""Authenticated direct-reference query-canvas consumer, never launch approval.

Required preprocess keys are raw_root, source_sha256, recipe_sha256,
weights_sha256 and path/SHA256 bindings enrollment_authority,
product_root_authority, phase_authority, records_authority, split_authority.
Enrollment bytes retain their original paths and pinned completed digest.
Root authority uses direct_reference_product_root_v1, original_root,
relocated_root and enrollment_manifest_sha256. It never rewrites the manifest.

Phase authority uses direct_reference_phase_v1, phase coordinate/overfit/smoke/
train, source/recipe/records/split/enrollment_manifest SHA256 identities and
authorized_image_ids with sorted unique, role-disjoint train/val lists.
Coordinate/overfit forbid val authorization. Smoke authorizes exactly 32 of
each role and requires externally hashed training_gates off/on. Train requires
these plus smoke_gates off/on. Training receipts use
direct_reference_training_gates_v1, the same identities plus weights_sha256,
reference_enabled, opened_roles=['train'], no_test_open/passed true,
coordinate_gate/model_gate passed true, and overfit n_images=16,
15<=n_correct<=16, 0<=image_loss<.15, 1<=updates<=500. Smoke receipts use
direct_reference_smoke_gates_v1, the same identities/arm, passed/no_test_open
true, n_images train=val=32, epochs_completed=1, replay passed/fresh_process
true. Numbers are finite and booleans are not numeric counts.

File authentication is not scientific validation or human training approval.
The runner owns gate production and transitions. Only build/load receipts are
accepted by the dataset; deserialize a table through load_registration_table,
not directly through json.load. ValueError denotes invalid authorities,
membership or content, while ordinary unsuccessful registration is retained
with context unavailable and mandatory referral. No test-role pixel access,
query annotation routing, lazy registration or median read in off is allowed.
Tables may contain an authorized superset; consumers open only selected rows.
Augmentation uses direct_reference_canvas_v1, query-only photometry and zone
jitter. Legacy numeric validation is reused after explicit semantic translation.
Metadata is authenticated at construction and reused while file fingerprints
are unchanged. Changed authority files must still match their pinned hashes.
The dataset table is a detached recursively immutable snapshot. Nested mutation
is rejected immediately, before any getter or pixel read, rather than detected
by rehashing the entire table per sample. Original content/authority hashes stay
unchanged. Manifest/preprocess mutation checks and per-file hashes remain active.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import random
import threading
from types import MappingProxyType

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import PairDataset, _normalize, _photometric
from .enrollment import _fit, _similarity, _warp, resolve_enrollment_bank
from .query import load_role_rows
from .rectify import to_canvas
from .register import _features, MIN_INLIERS, N_FEATURES, RATIO, RANSAC_PX
from ..pilot_config import resolve_augmentation

ENROLLMENT_SHA256 = "7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8"
WEIGHTS_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"
SCHEMA = "direct_reference_registration_v1"
IDENTITIES = ("source_sha256", "recipe_sha256", "records_sha256", "split_sha256", "enrollment_manifest_sha256")
_RNG_LOCK = threading.RLock()
_spec = importlib.util.spec_from_file_location(
    "direct_reference_product_verifier", Path(__file__).resolve().parents[3] / "scripts/verify_anchor_enrollment_products.py")
_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_verifier)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(value):
    path = _verifier._absolute(value)
    _verifier._no_symlinks(path)
    return path.resolve()


def _bytes(binding):
    _require(isinstance(binding, dict) and _verifier._sha256_value(binding.get("sha256")), "path/SHA256 authority required")
    path = _path(binding.get("path"))
    observed, content = _verifier._read_file(path, capture=True)
    _require(observed == binding["sha256"], f"SHA256 mismatch: {path}")
    return content


def _json(binding):
    value = _verifier._json(_bytes(binding))
    def finite(item):
        if isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
        elif isinstance(item, float):
            _require(math.isfinite(item), "nonfinite JSON number")
    finite(value)
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _gate(binding, identities, weights, enabled, smoke=False):
    gate = _json(binding)
    schema = "direct_reference_smoke_gates_v1" if smoke else "direct_reference_training_gates_v1"
    _require(gate.get("schema_version") == schema, "unsupported gate schema")
    _require(all(gate.get(key) == value for key, value in identities.items()), "gate identity mismatch")
    _require(gate.get("weights_sha256") == weights and gate.get("reference_enabled") is enabled, "gate weights/arm mismatch")
    _require(gate.get("passed") is True and gate.get("no_test_open") is True, "failed or unsafe gate")
    if smoke:
        counts = gate.get("n_images", {})
        _require(set(counts) == {"train", "val"} and all(type(value) is int and value == 32 for value in counts.values()), "smoke membership counts")
        _require(type(gate.get("epochs_completed")) is int and gate["epochs_completed"] == 1, "smoke epoch count")
        replay = gate.get("replay", {})
        _require(replay.get("passed") is True and replay.get("fresh_process") is True, "smoke replay failed")
    else:
        _require(gate.get("opened_roles") == ["train"], "training gate opened forbidden roles")
        _require(gate.get("coordinate_gate", {}).get("passed") is True and gate.get("model_gate", {}).get("passed") is True, "training prerequisite failed")
        probe = gate.get("overfit", {})
        _require(type(probe.get("n_images")) is int and probe["n_images"] == 16, "overfit image count")
        _require(type(probe.get("n_correct")) is int and 15 <= probe["n_correct"] <= 16, "overfit accuracy")
        _require(_number(probe.get("image_loss")) and 0 <= probe["image_loss"] < .15, "overfit loss")
        _require(type(probe.get("updates")) is int and 1 <= probe["updates"] <= 500, "overfit update count")


def _authorities(manifest, preprocess):
    _require(isinstance(preprocess, dict), "preprocess authorities required")
    enrollment = preprocess["enrollment_authority"]
    _require(enrollment["sha256"] == ENROLLMENT_SHA256, "unapproved enrollment authority")
    actual_manifest = _json(enrollment)
    _require(actual_manifest == manifest, "enrollment dictionary differs from authenticated bytes")
    identities = {"source_sha256": preprocess["source_sha256"], "recipe_sha256": preprocess["recipe_sha256"],
                  "records_sha256": preprocess["records_authority"]["sha256"],
                  "split_sha256": preprocess["split_authority"]["sha256"], "enrollment_manifest_sha256": ENROLLMENT_SHA256}
    _require(all(_verifier._sha256_value(value) for value in identities.values()), "invalid preprocessing identity")
    _require(preprocess["weights_sha256"] == WEIGHTS_SHA256, "unapproved original weights")
    phase = _json(preprocess["phase_authority"])
    _require(phase.get("schema_version") == "direct_reference_phase_v1" and phase.get("phase") in ("coordinate", "overfit", "smoke", "train"), "unsupported phase authority")
    _require(all(phase.get(key) == value for key, value in identities.items()), "phase identity mismatch")
    allowed = phase.get("authorized_image_ids")
    _require(isinstance(allowed, dict) and set(allowed) == {"train", "val"}, "explicit train/val authorization required")
    for identifiers in allowed.values():
        _require(isinstance(identifiers, list) and all(isinstance(value, str) and value for value in identifiers)
                 and identifiers == sorted(set(identifiers)), "authorized IDs must be sorted unique strings")
    _require(not set(allowed["train"]) & set(allowed["val"]), "authorized roles overlap")
    if phase["phase"] in ("coordinate", "overfit"):
        _require(not allowed["val"], "training-only phase forbids validation authorization")
    else:
        if phase["phase"] == "smoke":
            _require(all(len(identifiers) == 32 for identifiers in allowed.values()), "smoke requires exact 32 train/32 val")
        for name, smoke in (("training_gates", False), ("smoke_gates", True)):
            if smoke and phase["phase"] != "train": continue
            receipts = phase.get(name)
            _require(isinstance(receipts, dict) and set(receipts) == {"off", "on"}, f"both {name} authorities required")
            for arm, enabled in (("off", False), ("on", True)):
                _gate(receipts[arm], identities, WEIGHTS_SHA256, enabled, smoke)
    root = _json(preprocess["product_root_authority"])
    _require(root.get("schema_version") == "direct_reference_product_root_v1" and root.get("enrollment_manifest_sha256") == ENROLLMENT_SHA256, "invalid root authority")
    original = _path(root.get("original_root"))
    relocated = _path(root.get("relocated_root"))
    _require(str(original) == manifest.get("output_root"), "root authority does not bind original output root")
    records_content = _bytes(preprocess["records_authority"])
    split_content = _bytes(preprocess["split_authority"])
    official = {role: load_role_rows(BytesIO(records_content), BytesIO(split_content), role)
                for role in ("train", "val")}
    _bytes(preprocess["records_authority"])
    _bytes(preprocess["split_authority"])
    for role in ("train", "val"):
        _require(set(allowed[role]) <= set(official[role].image_id), "phase authorizes foreign role IDs")
    return dict(phase=phase, identities=identities, original=original, root=relocated, official=official,
                raw_root=_path(preprocess["raw_root"]))


def _rows(rows, context):
    _require(isinstance(rows, pd.DataFrame) and len(rows) > 0 and "image_id" in rows and not rows.image_id.duplicated().any(), "nonempty unique membership rows required")
    selected = []
    for row in rows.to_dict("records"):
        role, identifier = row.get("role"), row["image_id"]
        _require(role in ("train", "val") and identifier in context["phase"]["authorized_image_ids"][role], "query role/ID not phase-authorized")
        official = context["official"][role].set_index("image_id").loc[identifier]
        for key in ("path", "design_id", "label", "attack_family"):
            _require(key in row and row[key] == official[key] and not (key == "label" and isinstance(row[key], bool)), f"query {key} differs from official membership")
        if role == "train":
            _require(row.get("component_id") == official.component_id, "training component mismatch")
        relative = Path(row["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in row["path"], "unsafe raw query path")
        _path(str(context["raw_root"] / relative))
        selected.append(row)
    return selected


def _bank(manifest, row, context):
    component = row["component_id"] if row["role"] == "train" else None
    bank = resolve_enrollment_bank(manifest, row["design_id"], row["role"], component)
    contributors = bank.get("contributors", {})
    _require(set(contributors) == set(bank["support_ids"]) and set(bank["support_ids"]) <= set(manifest["eligible_ids"]), "unauthorized support dependencies")
    components = {item["component_id"] for item in contributors.values()}
    _require(components == set(bank["component_ids"]) and not components & set(manifest.get("forbidden_components", [])), "forbidden/transitive support components")
    train = context["official"]["train"].set_index("image_id")
    for identifier, contributor in contributors.items():
        _require(identifier in train.index and train.loc[identifier, "label"] == 0 and train.loc[identifier, "component_id"] == contributor["component_id"], "support not genuine official training provenance")
    return bank


def _products(bank, context):
    result = {}
    for name, binding in bank["products"].items():
        original = _path(binding["path"])
        expected = context["original"] / "fold0" / bank["design_id"] / bank["bank"] / f"{name}{'.json' if name in ('geometry', 'manifest') else '.png'}"
        _require(original == expected, "product path crosses bank/root")
        relocated = _path(str(context["root"] / original.relative_to(context["original"])))
        result[name] = dict(path=str(relocated), sha256=binding["sha256"])
    return result


def _image(binding, flags):
    image = cv2.imdecode(np.frombuffer(_bytes(binding), np.uint8), flags)
    _require(image is not None, f"cannot decode authenticated image: {binding['path']}")
    return image


def _plane(binding):
    image = _image(binding, cv2.IMREAD_GRAYSCALE)
    _require(image.shape == (512, 800) and np.isin(image, [0, 255]).all(), "context masks must be binary 512x800")
    return image


def _common(products, registration=True):
    for name in ("geometry", "manifest"):
        _json(products[name])
    target = None
    if registration:
        target = _image(products["target"], cv2.IMREAD_COLOR)
        _require(target.shape == (512, 800, 3), "bank target must be 512x800 BGR")
    return target, {name: _plane(products[name]) for name in ("target_valid", "valid", "zone")}


def _policy():
    return dict(version="direct_reference_query_canvas_v1", geometry="bank_similarity_v1", canvas_hw=[512, 800],
                opencv_version=cv2.__version__, n_features=N_FEATURES, ratio=RATIO, ransac_px=RANSAC_PX,
                min_inliers=MIN_INLIERS, implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _entry_binding(row, query_sha, bank, products, preprocess):
    value = dict(image_id=row["image_id"], design_id=row["design_id"], role=row["role"], path=row["path"],
                 query_sha256=query_sha, bank=bank["bank"], fold=bank["fold"], products=products,
                 preprocess_sha256=_digest(preprocess), policy=_policy())
    if row["role"] == "train": value["training_component"] = row["component_id"]
    return value


def _overlap(query_valid, masks, matrix):
    inverse = _similarity(np.linalg.inv(_similarity(matrix)))
    overlap = (query_valid > 0) & (_warp(masks["target_valid"], inverse, True) > 0) & (_warp(masks["valid"], inverse, True) > 0)
    return inverse, overlap.astype(np.uint8)


def _available(overlap):
    return bool(overlap.any() and cv2.resize(overlap, (100, 64), interpolation=cv2.INTER_NEAREST).any())


class _AuthenticatedTable(dict):
    def __init__(self, content, file_binding=None):
        super().__init__(copy.deepcopy(content))
        self.content_sha256 = _digest(content)
        self.file_binding = copy.deepcopy(file_binding)
        self.authority_sha256 = file_binding["sha256"] if file_binding else self.content_sha256


def _immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _immutable(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(child) for child in value)
    _require(value is None or type(value) in (str, int, float, bool), "table must contain JSON values")
    return value


@dataclass(frozen=True)
class _TableSnapshot(Mapping):
    content: Mapping
    content_sha256: str
    authority_sha256: str

    def __getitem__(self, key):
        return self.content[key]

    def __iter__(self):
        return iter(self.content)

    def __len__(self):
        return len(self.content)


def _validate_table(table, rows, manifest, preprocess, context):
    _require(table.get("schema_version") == SCHEMA and table.get("preprocess") == preprocess and table.get("policy") == _policy(), "table schema/policy/preprocessing mismatch")
    entries = table.get("entries")
    _require(isinstance(entries, dict) and bool(entries), "table entries required")
    for identifier, entry in entries.items():
        _require(isinstance(entry, dict) and entry.get("image_id") == identifier and entry.get("role") in ("train", "val"), "malformed registration entry")
        role = entry["role"]
        _require(identifier in context["phase"]["authorized_image_ids"][role], "table includes unauthorized query")
        official = context["official"][role].set_index("image_id").loc[identifier].to_dict()
        row = dict(official, image_id=identifier, role=role)
        bank = _bank(manifest, row, context)
        binding = _entry_binding(row, entry.get("query_sha256"), bank, _products(bank, context), preprocess)
        _require(_verifier._sha256_value(entry.get("query_sha256")) and all(entry.get(key) == value for key, value in binding.items()), "registration query/bank/product binding mismatch")
        _require(entry.get("key") == _digest(binding), "registration key mismatch")
        _require(type(entry.get("inliers")) is int and entry["inliers"] >= 0, "invalid inlier count")
        if entry.get("status") == "fit":
            _require(entry.get("reason") == "fit" and entry["inliers"] >= MIN_INLIERS, "invalid successful fit")
            _similarity(np.asarray(entry.get("query_canvas_to_bank"), np.float64))
        else:
            _require(entry.get("status") == "failed" and entry.get("reason") in ("registration_failed", "empty_overlap")
                     and entry.get("query_canvas_to_bank") is None, "invalid failed registration")
        _require(role == "train" or "training_component" not in entry, "evaluation cache contains component routing")
    for row in _rows(rows, context):
        _require(row["image_id"] in entries, "query missing from registration table")


def build_registration_table(rows: pd.DataFrame, raw_root: Path, enrollment_manifest: dict, *, preprocess: dict) -> dict:
    try:
        context = _authorities(enrollment_manifest, preprocess)
        _require(_path(str(raw_root)) == context["raw_root"], "raw root authority mismatch")
        selected = _rows(rows, context)
        report = _verifier.verify_products(preprocess["enrollment_authority"]["path"], ENROLLMENT_SHA256, output_root=context["root"])
        _require(report["passed"], f"enrollment product verification failed: {report['errors']}")
        entries = {}
        for row in selected:
            bank = _bank(enrollment_manifest, row, context)
            products = _products(bank, context)
            target, masks = _common(products)
            path = _path(str(context["raw_root"] / row["path"]))
            query_sha = _verifier._read_file(path)[0]
            raw = _image(dict(path=str(path), sha256=query_sha), cv2.IMREAD_COLOR)
            canvas, valid, _ = to_canvas(raw)
            binding = _entry_binding(row, query_sha, bank, products, preprocess)
            key = _digest(binding)
            inliers = 0
            try:
                matrix, inliers = _fit(_features(canvas), _features(target), int(key[:8], 16) & 0x7fffffff)
                _require(matrix is not None and type(inliers) is int and inliers >= MIN_INLIERS, "registration failed")
                matrix = _similarity(matrix)
                _, overlap = _overlap(valid, masks, matrix)
                reason = "fit" if _available(overlap) else "empty_overlap"
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                matrix, reason = None, "registration_failed"
                inliers = inliers if type(inliers) is int and inliers >= 0 else 0
            entries[row["image_id"]] = dict(binding, key=key, status="fit" if reason == "fit" else "failed", reason=reason,
                                             inliers=inliers, query_canvas_to_bank=matrix.tolist() if reason == "fit" else None)
        table = dict(schema_version=SCHEMA, preprocess=copy.deepcopy(preprocess), policy=_policy(), entries=entries)
        _validate_table(table, rows, enrollment_manifest, preprocess, context)
        return _AuthenticatedTable(table)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed direct-reference authority/table: {error}") from error


def load_registration_table(path: Path, expected_sha256: str, *, rows: pd.DataFrame, enrollment_manifest: dict, preprocess: dict) -> dict:
    try:
        context = _authorities(enrollment_manifest, preprocess)
        _rows(rows, context)
        binding = dict(path=str(path), sha256=expected_sha256)
        table = _json(binding)
        _validate_table(table, rows, enrollment_manifest, preprocess, context)
        report = _verifier.verify_products(preprocess["enrollment_authority"]["path"], ENROLLMENT_SHA256, output_root=context["root"])
        _require(report["passed"], "enrollment product verification failed")
        for row in _rows(rows, context):
            _bytes(dict(path=str(context["raw_root"] / row["path"]), sha256=table["entries"][row["image_id"]]["query_sha256"]))
        return _AuthenticatedTable(table, binding)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed direct-reference authority/table: {error}") from error


@contextmanager
def _isolated_rng(generator):
    with _RNG_LOCK:
        previous = random.getstate()
        random.setstate(generator.getstate())
        try:
            yield
        finally:
            generator.setstate(random.getstate())
            random.setstate(previous)


def _augmentation(supplied):
    translated = dict(supplied)
    _require(translated.pop("preset", "direct_reference_canvas_v1") == "direct_reference_canvas_v1", "unsupported direct-reference augmentation preset")
    _require(translated.pop("reference_photometric", False) is False, "reference photometry is forbidden")
    _require(translated.pop("jitter_zone", True) is True, "zone must follow coherent geometry and jitter")
    translated.update(preset="legacy_query_v1", jitter_zone=False)
    result = resolve_augmentation("D_running_pilot", translated)
    result.update(preset="direct_reference_canvas_v1", reference_photometric=False, jitter_zone=True)
    return result


def _fingerprint(binding):
    status = _path(binding["path"]).stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _authority_bindings(preprocess, phase):
    bindings = [preprocess[name] for name in ("enrollment_authority", "product_root_authority", "phase_authority", "records_authority", "split_authority")]
    for name in ("training_gates", "smoke_gates"):
        bindings.extend(phase.get(name, {}).values())
    return bindings


class DirectReferenceDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, raw_root: Path, enrollment_manifest: dict, registration_table: dict, *, reference_enabled: bool, train: bool, augment: dict):
        try:
            _require(type(reference_enabled) is bool and type(train) is bool, "dataset mode booleans required")
            _require(isinstance(registration_table, _AuthenticatedTable), "use authenticated table builder/loader, not a plain dictionary")
            registration_table = copy.deepcopy(registration_table)
            _require(_digest(registration_table) == registration_table.content_sha256, "authenticated table was mutated")
            if registration_table.file_binding: _bytes(registration_table.file_binding)
            preprocess = registration_table["preprocess"]
            context = _authorities(enrollment_manifest, preprocess)
            _require(_path(str(raw_root)) == context["raw_root"], "raw root authority mismatch")
            self.rows = _rows(rows, context)
            _require(all(row["role"] == ("train" if train else "val") for row in self.rows), "dataset train/validation role mismatch")
            _validate_table(registration_table, rows, enrollment_manifest, preprocess, context)
            self._table = _TableSnapshot(_immutable(registration_table), registration_table.content_sha256,
                                         registration_table.authority_sha256)
            self.manifest, self.preprocess = copy.deepcopy(enrollment_manifest), copy.deepcopy(preprocess)
            self.raw_root, self.context = context["raw_root"], context
            self.reference_enabled, self.train = reference_enabled, train
            supplied = dict(augment)
            seed = supplied.pop("seed", 42)
            _require(type(seed) is int, "augmentation seed must be integer")
            self.augment = _augmentation(supplied)
            self.crop, self.affine_px = None, self.augment["affine_px"]
            self.geometry_rng, self.photometry_rng = random.Random(seed), random.Random(seed + 1)
            self.metadata_sha256 = _digest(dict(manifest=self.manifest, preprocess=self.preprocess))
            self.authority_files = copy.deepcopy(_authority_bindings(preprocess, context["phase"]))
            if registration_table.file_binding: self.authority_files.append(copy.deepcopy(registration_table.file_binding))
            self.authority_fingerprints = []
            for binding in self.authority_files:
                before = _fingerprint(binding)
                _bytes(binding)
                _require(before == _fingerprint(binding), "authority changed during authentication")
                self.authority_fingerprints.append(before)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"malformed direct-reference dataset: {error}") from error

    def __len__(self):
        return len(self.rows)

    @property
    def table(self):
        return self._table

    def __getitem__(self, index):
        _require(_digest(dict(manifest=self.manifest, preprocess=self.preprocess)) == self.metadata_sha256, "authenticated metadata was mutated")
        for position, binding in enumerate(self.authority_files):
            observed = _fingerprint(binding)
            if observed != self.authority_fingerprints[position]:
                _bytes(binding)
                _require(observed == _fingerprint(binding), "authority changed during authentication")
                self.authority_fingerprints[position] = observed
        context = self.context
        row = self.rows[index]
        _rows(pd.DataFrame([row]), context)
        entry = self.table["entries"][row["image_id"]]
        products = _products(_bank(self.manifest, row, context), context)
        _, masks = _common(products, registration=False)
        raw = _image(dict(path=str(self.raw_root / row["path"]), sha256=entry["query_sha256"]), cv2.IMREAD_COLOR)
        query, query_valid, _ = to_canvas(raw)
        reference = None
        overlap, zone = np.zeros((512, 800), np.uint8), np.zeros((512, 800), np.uint8)
        reason = entry["reason"]
        if entry["status"] == "fit":
            inverse, overlap = _overlap(query_valid, masks, np.asarray(entry["query_canvas_to_bank"]))
            zone = (_warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap
            if self.reference_enabled and _available(overlap):
                median = _image(products["median"], cv2.IMREAD_COLOR)
                _require(median.shape == (512, 800, 3), "median must be 512x800 BGR")
                reference = _warp(median, inverse)
        if self.train and self.augment["enabled"]:
            with _isolated_rng(self.geometry_rng):
                matrix, window = PairDataset._geometry(self, 512, 800)
                query = PairDataset._warp(query, matrix, window, False)
                zone = PairDataset._warp(zone, matrix, window, True)
                overlap = PairDataset._warp(overlap, matrix, window, True)
                if reference is not None: reference = PairDataset._warp(reference, matrix, window, False)
                if self.augment["jitter_px"] and random.random() < self.augment["jitter_p"]:
                    jitter = np.float32([[1, 0, random.randint(-self.augment["jitter_px"], self.augment["jitter_px"])],
                                        [0, 1, random.randint(-self.augment["jitter_px"], self.augment["jitter_px"])]])
                    query = cv2.warpAffine(query, jitter, (800, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    if reference is not None:
                        reference = cv2.warpAffine(reference, jitter, (800, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    zone = cv2.warpAffine(zone, jitter, (800, 512), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    overlap = cv2.warpAffine(overlap, jitter, (800, 512), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if self.augment["photometric"]:
                with _isolated_rng(self.photometry_rng): query = _photometric(query, .2)
        available = _available(overlap)
        if not available:
            zone.fill(0)
            overlap.fill(0)
            if reason == "fit": reason = "empty_overlap"
        zone *= overlap
        result = dict(query=torch.from_numpy(np.ascontiguousarray(np.concatenate([_normalize(query), zone[None].astype(np.float32)]))),
                      overlap=torch.from_numpy(overlap[None].astype(np.float32)), context_available=torch.tensor(available, dtype=torch.bool),
                      label=int(row["label"]), meta=dict(image_id=row["image_id"], design_id=row["design_id"], role=row["role"],
                          attack_family=row["attack_family"], fold=entry["fold"], bank=entry["bank"], registration_key=entry["key"],
                          registration_table_sha256=self.table.authority_sha256, query_sha256=entry["query_sha256"],
                          products=copy.deepcopy(products), availability_reason=reason, requires_referral=not available))
        if self.reference_enabled:
            result["reference"] = torch.from_numpy(np.ascontiguousarray(_normalize(reference))) if available else torch.zeros((3, 512, 800), dtype=torch.float32)
        return result
