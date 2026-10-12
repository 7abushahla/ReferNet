"""Explicit Protocol A fold-zero evaluation, separate from training phase authority.

All five input authorities are path/SHA256 bindings. Evaluation JSON requires
schema_version=protocol_a_evaluation_v1, protocol=A, integer fold=0,
records_sha256, split_sha256 (CSV bytes), enrollment_manifest_sha256,
test_image_ids (the exact sorted official test membership), literal
source_validation_selected=true, and checkpoints={off,on} with path/SHA256
bindings to the selected checkpoint files. These files are hashed, not loaded.
The externally authenticated evaluation document authorizes the transition
after both full runs; this consumer cannot independently prove checkpoint
selection or completion from a hash and does not invent a training receipt.

Root authority retains direct_reference_product_root_v1 and the original
manifest bytes. Support integrity and canonical relocation use the original
verifier. Test annotations/components never route registration or tensors.
Registration has its own protocol_a_evaluation_registration_v1 schema, opens
queries only after authorization, uses full enrollment banks, and retains
ordinary fit failures as mandatory referrals. No augmentation or lazy fitting.
OFF and ON share an authenticated registration table and identical query input.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import direct_reference as direct
from .enrollment import _table_hash
from .query import _RECORD_COLUMNS, _SPLIT_COLUMNS, _validate_rows

ENROLLMENT_SHA256 = "7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8"
ROLE_COUNTS = {"train": 1778, "val": 222, "test": 222}
SCHEMA = "protocol_a_evaluation_registration_v1"


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _checkpoint(binding):
    direct._require(isinstance(binding, dict) and direct._verifier._sha256_value(binding.get("sha256")), "checkpoint path/SHA256 binding required")
    path = direct._path(binding.get("path"))
    observed = direct._verifier._read_file(path)[0]
    direct._require(observed == binding["sha256"], "selected checkpoint SHA256 mismatch")


@dataclass(frozen=True)
class _EvaluationContext:
    payload: bytes
    sha256: str

    @property
    def rows(self):
        return pd.DataFrame(_state(self)["rows"])


@dataclass(frozen=True)
class _EvaluationRegistration:
    payload: bytes
    sha256: str
    path: str


def _state(context):
    direct._require(isinstance(context, _EvaluationContext) and _sha(context.payload) == context.sha256, "authenticated evaluation context required")
    state = json.loads(context.payload)
    for item in state["authorities"]:
        binding = item["binding"]
        observed = direct._fingerprint(binding)
        if list(observed) != item["fingerprint"]:
            if item["checkpoint"]:
                _checkpoint(binding)
            else:
                direct._bytes(binding)
            direct._require(observed == direct._fingerprint(binding), "authority changed during authentication")
    return state


def _membership(records_content, split_content):
    records = pd.read_csv(BytesIO(records_content), keep_default_na=False)
    split = pd.read_csv(BytesIO(split_content), keep_default_na=False)
    _validate_rows(records, _RECORD_COLUMNS, "records")
    _validate_rows(split, _SPLIT_COLUMNS, "split")
    direct._require("dataset" in records and "protocol" in split and "ref_group" in split, "official Protocol A metadata required")
    direct._require(split.protocol.eq("A").all() and split.ref_group.eq("it0").all(), "official Protocol A fold-zero membership required")
    direct._require(split.role.value_counts().to_dict() == ROLE_COUNTS, "official membership counts mismatch")
    records = records.loc[records.dataset == "sidtd"]
    direct._require(set(records.image_id) == set(split.image_id), "official membership discrepancy")
    joined = split.merge(records[list(_RECORD_COLUMNS)], on="image_id", validate="one_to_one", suffixes=("", "_record"))
    for name in ("label", "design_id", "component_id"):
        direct._require(joined[name].eq(joined[f"{name}_record"]).all(), f"official {name} mismatch")
    joined["label"] = joined.label.astype(int)
    return joined, split


def load_evaluation_context(records_binding, split_binding, enrollment_binding, root_binding, raw_root, *, evaluation_binding):
    """Authenticate explicit test authority and support without decoding images."""
    try:
        records_binding, split_binding, enrollment_binding, root_binding, evaluation_binding = copy.deepcopy(
            (records_binding, split_binding, enrollment_binding, root_binding, evaluation_binding))
        evaluation = direct._json(evaluation_binding)
        direct._require(evaluation.get("schema_version") == "protocol_a_evaluation_v1"
                        and evaluation.get("protocol") == "A" and type(evaluation.get("fold")) is int
                        and evaluation["fold"] == 0 and evaluation.get("source_validation_selected") is True,
                        "explicit validation-selected Protocol A fold-zero evaluation binding required")
        direct._require(enrollment_binding["sha256"] == ENROLLMENT_SHA256, "unapproved original enrollment authority")
        identities = dict(records_sha256=records_binding["sha256"], split_sha256=split_binding["sha256"],
                          enrollment_manifest_sha256=ENROLLMENT_SHA256)
        direct._require(all(evaluation.get(key) == value for key, value in identities.items()), "evaluation metadata identities mismatch")
        checkpoints = evaluation.get("checkpoints")
        direct._require(isinstance(checkpoints, dict) and set(checkpoints) == {"off", "on"}, "both selected checkpoint bindings required")
        for binding in checkpoints.values():
            _checkpoint(binding)
        records_content, split_content = direct._bytes(records_binding), direct._bytes(split_binding)
        joined, split = _membership(records_content, split_content)
        tests = joined.loc[joined.role == "test"].sort_values("image_id")
        direct._require(evaluation.get("test_image_ids") == tests.image_id.tolist(), "exact sorted official test IDs required")
        manifest = direct._json(enrollment_binding)
        direct._require(type(manifest.get("fold")) is int and manifest["fold"] == 0
                        and manifest.get("official_fold_binding") == dict(protocol="A", fold=0, ref_group="it0"), "original fold-zero enrollment required")
        direct._require(manifest.get("official_split_hash_format") == "canonical_json_rows_v1"
                        and manifest.get("official_split_sha256") == _table_hash(split), "enrollment canonical split binding mismatch")
        relocation = direct._json(root_binding)
        direct._require(relocation.get("schema_version") == "direct_reference_product_root_v1"
                        and relocation.get("enrollment_manifest_sha256") == ENROLLMENT_SHA256, "invalid root relocation authority")
        original = direct._path(relocation.get("original_root"))
        root = direct._path(relocation.get("relocated_root"))
        direct._require(str(original) == manifest.get("output_root"), "original root binding mismatch")
        raw = direct._path(raw_root)
        geometry_context = dict(original=original, root=root, official={"train": joined.loc[joined.role == "train"]})
        rows, banks = [], {}
        for row in tests.to_dict("records"):
            relative = Path(row["path"])
            direct._require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in row["path"], "unsafe query path")
            direct._path(raw / relative)
            selected = {key: row[key] for key in ("image_id", "path", "design_id", "label", "attack_family", "role")}
            bank = direct._bank(manifest, selected, geometry_context)
            banks[row["design_id"]] = dict(bank=bank["bank"], fold=bank["fold"], products=direct._products(bank, geometry_context))
            rows.append(selected)
        report = direct._verifier.verify_products(enrollment_binding["path"], ENROLLMENT_SHA256, output_root=root)
        direct._require(report["passed"], f"support product integrity failed: {report['errors']}")
        authorities = []
        for binding in (records_binding, split_binding, enrollment_binding, root_binding, evaluation_binding, *checkpoints.values()):
            before = direct._fingerprint(binding)
            is_checkpoint = binding in checkpoints.values()
            if is_checkpoint:
                _checkpoint(binding)
            else:
                direct._bytes(binding)
            direct._require(before == direct._fingerprint(binding), "authority changed during context authentication")
            authorities.append(dict(binding=binding, fingerprint=list(before), checkpoint=is_checkpoint))
        state = dict(rows=rows, banks=banks, raw_root=str(raw), original_root=str(original), product_root=str(root),
                     identities=identities, evaluation_sha256=evaluation_binding["sha256"], policy=direct._policy(), authorities=authorities)
        payload = _encode(state)
        return _EvaluationContext(payload, _sha(payload))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed evaluation authority: {error}") from error


def _entry_binding(state, row, query_sha256):
    bank = state["banks"][row["design_id"]]
    return dict(image_id=row["image_id"], path=row["path"], design_id=row["design_id"], role="test",
                query_sha256=query_sha256, bank=bank["bank"], fold=bank["fold"], products=bank["products"], policy=state["policy"])


def _validate(context, document):
    state = _state(context)
    direct._require(document.get("schema_version") == SCHEMA and document.get("context_sha256") == context.sha256
                    and document.get("policy") == state["policy"] == direct._policy(), "evaluation registration schema/context/policy mismatch")
    entries = document.get("entries")
    direct._require(isinstance(entries, dict) and set(entries) == {row["image_id"] for row in state["rows"]}, "exact evaluation registration membership required")
    for row in state["rows"]:
        entry = entries[row["image_id"]]
        direct._require(isinstance(entry, dict) and direct._verifier._sha256_value(entry.get("query_sha256")), "invalid query SHA256")
        expected = _entry_binding(state, row, entry["query_sha256"])
        direct._require(all(entry.get(key) == value for key, value in expected.items())
                        and entry.get("key") == direct._digest(expected), "evaluation registration binding mismatch")
        direct._require("training_component" not in entry and "component_id" not in entry and "annotation" not in entry, "query provenance routing forbidden")
        direct._require(type(entry.get("inliers")) is int and entry["inliers"] >= 0, "invalid registration inliers")
        if entry.get("status") == "fit":
            direct._require(entry.get("reason") == "fit" and entry["inliers"] >= direct.MIN_INLIERS, "invalid registration success")
            direct._similarity(np.asarray(entry.get("query_canvas_to_bank"), np.float64))
        else:
            direct._require(entry.get("status") == "failed" and entry.get("reason") in ("registration_failed", "empty_overlap")
                            and entry.get("query_canvas_to_bank") is None, "invalid registration failure")
    return state


def build_evaluation_registration(context, output_path):
    """Fit each authorized test query once, without annotations or median decoding."""
    state = _state(context)
    output = direct._path(output_path)
    direct._require(not output.exists() and not any(output.is_relative_to(Path(state[name])) for name in ("raw_root", "product_root", "original_root")), "new registration path outside input roots required")
    entries = {}
    for row in state["rows"]:
        products = state["banks"][row["design_id"]]["products"]
        target, masks = direct._common(products)
        path = direct._path(Path(state["raw_root"]) / row["path"])
        query_sha = direct._verifier._read_file(path)[0]
        raw = direct._image(dict(path=str(path), sha256=query_sha), cv2.IMREAD_COLOR)
        canvas, valid, _ = direct.to_canvas(raw)
        expected = _entry_binding(state, row, query_sha)
        key = direct._digest(expected)
        inliers = 0
        try:
            matrix, inliers = direct._fit(direct._features(canvas), direct._features(target), int(key[:8], 16) & 0x7fffffff)
            direct._require(matrix is not None and type(inliers) is int and inliers >= direct.MIN_INLIERS, "registration failed")
            matrix = direct._similarity(matrix)
            _, overlap = direct._overlap(valid, masks, matrix)
            reason = "fit" if direct._available(overlap) else "empty_overlap"
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            matrix, reason = None, "registration_failed"
            inliers = inliers if type(inliers) is int and inliers >= 0 else 0
        entries[row["image_id"]] = dict(expected, key=key, status="fit" if reason == "fit" else "failed", reason=reason,
            inliers=inliers, query_canvas_to_bank=matrix.tolist() if reason == "fit" else None)
    document = dict(schema_version=SCHEMA, context_sha256=context.sha256, policy=state["policy"], entries=entries)
    _validate(context, document)
    payload = _encode(document)
    with output.open("xb") as stream:
        stream.write(payload)
    return _EvaluationRegistration(payload, _sha(payload), str(output))


def load_evaluation_registration(context, binding):
    payload = direct._bytes(binding)
    document = direct._verifier._json(payload)
    _validate(context, document)
    state = _state(context)
    for row in state["rows"]:
        direct._bytes(dict(path=str(Path(state["raw_root"]) / row["path"]), sha256=document["entries"][row["image_id"]]["query_sha256"]))
    return _EvaluationRegistration(payload, _sha(payload), str(direct._path(binding["path"])))


class ProtocolAEvaluationDataset(Dataset):
    def __init__(self, context, table, *, reference_enabled):
        direct._require(type(reference_enabled) is bool, "reference_enabled must be boolean")
        direct._require(isinstance(table, _EvaluationRegistration) and _sha(table.payload) == table.sha256, "authenticated evaluation registration required")
        direct._bytes(dict(path=table.path, sha256=table.sha256))
        self._state = _validate(context, direct._verifier._json(table.payload))
        self._context, self._table = context, table
        self.reference_enabled = reference_enabled

    def __len__(self):
        return len(self._state["rows"])

    def __getitem__(self, index):
        state = _state(self._context)
        direct._bytes(dict(path=self._table.path, sha256=self._table.sha256))
        document = direct._verifier._json(self._table.payload)
        row = state["rows"][index]
        entry = document["entries"][row["image_id"]]
        products = state["banks"][row["design_id"]]["products"]
        _, masks = direct._common(products, registration=False)
        raw = direct._image(dict(path=str(Path(state["raw_root"]) / row["path"]), sha256=entry["query_sha256"]), cv2.IMREAD_COLOR)
        query, query_valid, _ = direct.to_canvas(raw)
        overlap, zone = np.zeros((512, 800), np.uint8), np.zeros((512, 800), np.uint8)
        reference, reason = None, entry["reason"]
        if entry["status"] == "fit":
            inverse, overlap = direct._overlap(query_valid, masks, np.asarray(entry["query_canvas_to_bank"]))
            zone = (direct._warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap
            if self.reference_enabled and direct._available(overlap):
                median = direct._image(products["median"], cv2.IMREAD_COLOR)
                direct._require(median.shape == (512, 800, 3), "median must be 512x800 BGR")
                reference = direct._warp(median, inverse)
        available = direct._available(overlap)
        if not available:
            zone.fill(0)
            overlap.fill(0)
            if reason == "fit":
                reason = "empty_overlap"
        zone *= overlap
        result = dict(query=torch.from_numpy(np.ascontiguousarray(np.concatenate([direct._normalize(query), zone[None].astype(np.float32)]))),
            overlap=torch.from_numpy(overlap[None].astype(np.float32)), context_available=torch.tensor(available, dtype=torch.bool),
            label=int(row["label"]), meta=dict(image_id=row["image_id"], design_id=row["design_id"], role="test",
                attack_family=row["attack_family"], bank="enrollment", fold=0, registration_key=entry["key"],
                registration_table_sha256=self._table.sha256, query_sha256=entry["query_sha256"],
                products=products, availability_reason=reason, requires_referral=not available))
        if self.reference_enabled:
            result["reference"] = torch.from_numpy(np.ascontiguousarray(direct._normalize(reference))) if available else torch.zeros((3, 512, 800), dtype=torch.float32)
        return result
