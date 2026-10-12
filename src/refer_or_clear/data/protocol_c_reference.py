"""C-only evaluation of authenticated, source-selected A OFF/ON checkpoints.

An externally hash-bound protocol_c_evaluation_v1 document must declare
source_protocol=A, source_fold=0, source_validation_selected=true, the five
metadata SHA256 identities, exact remaining authorized_image_ids, and both
selected checkpoint bindings. Checkpoint bytes are hashed, never loaded here.
Hashes do not prove source training completion or selection. The future checked
driver must validate both A parent/replay chains before issuing this authority.

Only the three headline target designs' full C enrollment banks are consumed.
Optional protocol_c_product_root_v1 relocation binds original/relocated roots
without rewriting the producer manifest. No target annotations or provenance
route query registration/tensors. No training, augmentation or lazy fitting.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..protocol_c_contract import load_protocol_c_contract, load_target_enrollment
from . import direct_reference as direct
from .protocol_c_enrollment import GEOMETRY_VERSION, PRODUCTS, make_protocol_c_enrollment_manifest

SCHEMA = "protocol_c_registration_v1"
DEPLOYMENT = "locked_a_selected_models_no_target_training"


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _policy():
    value = direct._policy()
    value.update(version="protocol_c_query_canvas_v1", protocol="C",
                 implementation_sha256=direct._digest(dict(consumer=Path(__file__).read_text(), original=value["implementation_sha256"])))
    return value


def _checkpoint(binding):
    direct._require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
                    and direct._verifier._sha256_value(binding["sha256"]), "selected checkpoint path/SHA256 required")
    path = direct._path(binding["path"])
    direct._require(binding["path"] == str(path) and direct._verifier._read_file(path)[0] == binding["sha256"],
                    "selected checkpoint SHA256/canonical path mismatch")


@dataclass(frozen=True)
class ReferenceContext:
    payload: bytes
    sha256: str

    def rows(self, role="test"):
        direct._require(role == "test", "C consumer is evaluation-only")
        return pd.DataFrame(_state(self)["rows"])


@dataclass(frozen=True)
class RegistrationTable:
    payload: bytes
    sha256: str
    path: str


def _state(context):
    direct._require(isinstance(context, ReferenceContext) and _sha(context.payload) == context.sha256,
                    "authenticated immutable C context required")
    state = direct._verifier._json(context.payload)
    direct._require(state["policy"] == _policy(), "C consumer source/policy changed")
    for authority in state["authorities"]:
        binding = authority["binding"]
        observed = direct._fingerprint(binding)
        if list(observed) != authority["fingerprint"]:
            if authority["checkpoint"]:
                _checkpoint(binding)
            else:
                direct._bytes(binding)
            direct._require(observed == direct._fingerprint(binding), "authority changed during C authentication")
    return state


def _products(bank, expected, manifest, original, root):
    direct._require(bank.get("protocol") == "C" and bank.get("bank") == "enrollment"
                    and not {"fold", "group"} & set(bank)
                    and all(bank.get(name) == value for name, value in expected.items()), "C full-bank metadata mismatch")
    contributors = bank.get("contributors", {})
    direct._require(set(contributors) == set(expected["support_ids"]), "exact C bank contributors required")
    for identifier, contributor in contributors.items():
        source = manifest["authorized_contributors"][identifier]
        direct._require(all(contributor.get(name) == value for name, value in source.items())
                        and all(direct._verifier._sha256_value(contributor.get(name)) for name in ("raw_sha256", "layout_sha256")),
                        "C contributor metadata/hash binding mismatch")
    registered = bank.get("registered_ids")
    direct._require(isinstance(registered, list) and registered and len(set(registered)) == len(registered)
                    and set(registered) <= set(contributors), "C registered support membership mismatch")
    declared = bank.get("products", {})
    direct._require(set(declared) == PRODUCTS, "exact nine C products required")
    products = {}
    for name, binding in declared.items():
        suffix = ".json" if name in ("geometry", "manifest") else ".png"
        expected_path = original / "groupC" / bank["design_id"] / "enrollment" / (name + suffix)
        direct._require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
                        and binding["path"] == str(expected_path) and direct._path(binding["path"]) == expected_path,
                        "C product crosses canonical bank/root")
        products[name] = dict(path=str(root / expected_path.relative_to(original)), sha256=binding["sha256"])
        direct._bytes(products[name])
    geometry = direct._json(products["geometry"])
    identities = dict(schema_version="protocol_c_enrollment_v1", protocol="C", geometry_version=GEOMETRY_VERSION,
        design_id=bank["design_id"], bank="enrollment", seed_slot=0, canvas_hw=[512, 800],
        support_ids=expected["support_ids"], contributors=contributors, registered_ids=registered,
        source_split_sha256=manifest["source_split_sha256"], target_split_sha256=manifest["target_split_sha256"],
        target_enrollment_sha256=manifest["target_enrollment_authority"]["sha256"], source=manifest["source"])
    direct._require(all(type(geometry.get(name)) is type(value) and geometry[name] == value for name, value in identities.items())
                    and not {"fold", "group", "official_split_sha256"} & set(geometry)
                    and geometry.get("axes") == "x=column,y=row; native_to_bank=canvas_to_bank@native_to_canvas"
                    and geometry.get("valid_interpolation") == "nearest" and geometry.get("rgb_interpolation") == "linear"
                    and geometry.get("registration", {}).get("seed_slot_meaning") == "protocol_c_fixed_geometry_slot_not_fold",
                    "C geometry dependency identity mismatch")
    dependency = direct._json(products["manifest"])
    direct._require({name: value for name, value in dependency.items() if name != "output_sha256"} == geometry
                    and dependency.get("output_sha256") == {name: value["sha256"] for name, value in products.items() if name != "manifest"},
                    "C product dependency hashes/geometry mismatch")
    return products


def load_reference_context(records_csv, source_split_csv, target_split_csv, *, records_sha256,
                           target_binding, enrollment_binding, raw_root, evaluation_binding, root_binding=None):
    """Authenticate source selection and C target support before query access."""
    try:
        target_binding, enrollment_binding, evaluation_binding, root_binding = copy.deepcopy(
            (target_binding, enrollment_binding, evaluation_binding, root_binding))
        evaluation = direct._json(evaluation_binding)
        direct._require(evaluation.get("schema_version") == "protocol_c_evaluation_v1" and evaluation.get("protocol") == "C"
                        and evaluation.get("deployment") == DEPLOYMENT and evaluation.get("source_protocol") == "A"
                        and type(evaluation.get("source_fold")) is int and evaluation["source_fold"] == 0
                        and evaluation.get("source_validation_selected") is True and not {"fold", "group"} & set(evaluation),
                        "explicit C authorization of source-selected A checkpoints required")
        checkpoints = evaluation.get("checkpoints")
        direct._require(isinstance(checkpoints, dict) and set(checkpoints) == {"off", "on"}, "both selected A checkpoint bindings required")
        for binding in checkpoints.values():
            _checkpoint(binding)
        direct._require(checkpoints["off"]["path"] != checkpoints["on"]["path"], "distinct selected OFF/ON checkpoints required")
        contract = load_protocol_c_contract(records_csv, source_split_csv, target_split_csv, records_sha256=records_sha256)
        target = load_target_enrollment(contract, target_binding["path"], expected_sha256=target_binding["sha256"])
        identities = dict(protocol="C", records_sha256=contract.records_sha256, source_split_sha256=contract.source_split_sha256,
            target_split_sha256=contract.target_split_sha256, target_enrollment_sha256=target_binding["sha256"],
            enrollment_manifest_sha256=enrollment_binding["sha256"])
        direct._require(all(evaluation.get(name) == value for name, value in identities.items())
                        and evaluation.get("authorized_image_ids") == sorted(target.query_ids), "C authority metadata/remaining membership mismatch")
        raw = direct._path(raw_root)
        manifest = direct._json(enrollment_binding)
        expected = make_protocol_c_enrollment_manifest(contract, target, raw)
        for name, value in expected.items():
            if name not in ("banks", "contributors"):
                direct._require(manifest.get(name) == value, f"C enrollment {name} mismatch")
        direct._require(not {"fold", "group"} & set(manifest) and set(manifest["banks"]) == set(contract.target_designs)
                        and set(manifest["contributors"]) == set(expected["contributors"]), "exact C headline support authority required")
        original = direct._path(manifest["output_root"])
        root = original
        if root_binding is not None:
            relocation = direct._json(root_binding)
            direct._require(relocation.get("schema_version") == "protocol_c_product_root_v1" and relocation.get("protocol") == "C"
                            and relocation.get("enrollment_manifest_sha256") == enrollment_binding["sha256"]
                            and relocation.get("original_root") == str(original), "C product relocation authority mismatch")
            root = direct._path(relocation["relocated_root"])
        direct._require(not root.is_relative_to(raw) and not raw.is_relative_to(root), "C product/raw roots overlap")
        validation = dict(manifest, authorized_contributors=expected["contributors"])
        banks = {}
        for design in contract.target_designs:
            declared = manifest["banks"][design]
            direct._require(set(declared) == {"enrollment"}, "C target requires only full enrollment bank")
            bank = declared["enrollment"]
            direct._require(bank.get("contributors") == {identifier: manifest["contributors"][identifier] for identifier in bank["support_ids"]},
                            "C top-level/bank contributor discrepancy")
            banks[design] = dict(bank="enrollment", products=_products(bank, expected["banks"][design]["enrollment"], validation, original, root))
        records_binding = dict(path=str(direct._path(records_csv)), sha256=contract.records_sha256)
        records = pd.read_csv(BytesIO(direct._bytes(records_binding)), keep_default_na=False).set_index("image_id")
        rows = []
        members = {row.image_id: row for row in contract.target_test}
        for identifier in sorted(target.query_ids):
            metadata = records.loc[identifier]
            member = members[identifier]
            relative = Path(metadata["path"])
            direct._require(not relative.is_absolute() and relative.parts and relative.parts[0] == "fantasyid"
                            and ".." not in relative.parts and "\\" not in metadata["path"], "unsafe C target query path")
            direct._path(raw / relative)
            rows.append(dict(image_id=identifier, path=metadata["path"], design_id=member.design_id,
                             label=member.label, attack_family=metadata["attack_family"], role="test"))
        bindings = [records_binding, dict(path=str(direct._path(source_split_csv)), sha256=contract.source_split_sha256),
            dict(path=str(direct._path(target_split_csv)), sha256=contract.target_split_sha256), target_binding, enrollment_binding, evaluation_binding]
        if root_binding is not None:
            bindings.append(root_binding)
        bindings.extend(binding for bank in banks.values() for binding in bank["products"].values())
        bindings.extend(checkpoints.values())
        authorities = []
        for binding in bindings:
            before = direct._fingerprint(binding)
            is_checkpoint = binding in checkpoints.values()
            _checkpoint(binding) if is_checkpoint else direct._bytes(binding)
            direct._require(before == direct._fingerprint(binding), "C authority changed during context authentication")
            authorities.append(dict(binding=binding, fingerprint=list(before), checkpoint=is_checkpoint))
        state = dict(rows=rows, banks=banks, raw_root=str(raw), product_root=str(root), original_root=str(original),
            identities=identities, evaluation_sha256=evaluation_binding["sha256"], policy=_policy(), authorities=authorities)
        payload = _encode(state)
        return ReferenceContext(payload, _sha(payload))
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"malformed C evaluation authority: {error}") from error


def _entry_binding(state, row, query_sha256):
    return dict(image_id=row["image_id"], path=row["path"], design_id=row["design_id"], role="test", protocol="C",
        query_sha256=query_sha256, bank="enrollment", products=state["banks"][row["design_id"]]["products"],
        identities=state["identities"], evaluation_sha256=state["evaluation_sha256"], policy=state["policy"])


def _validate(context, document):
    state = _state(context)
    direct._require(document.get("schema_version") == SCHEMA and document.get("context_sha256") == context.sha256
                    and document.get("policy") == state["policy"] and not {"fold", "group"} & set(document), "C registration schema/context/policy mismatch")
    entries = document.get("entries")
    direct._require(isinstance(entries, dict) and set(entries) == {row["image_id"] for row in state["rows"]}, "exact remaining C query membership required")
    for row in state["rows"]:
        entry = entries[row["image_id"]]
        direct._require(isinstance(entry, dict) and direct._verifier._sha256_value(entry.get("query_sha256")), "invalid C query hash")
        expected = _entry_binding(state, row, entry["query_sha256"])
        direct._require(all(entry.get(name) == value for name, value in expected.items()) and entry.get("key") == direct._digest(expected),
                        "C registration query/bank binding mismatch")
        direct._require(not {"fold", "group", "annotation", "component_id", "identity_id", "source_id", "donor_id", "training_component"} & set(entry),
                        "C query annotation/provenance routing forbidden")
        direct._require(type(entry.get("inliers")) is int and entry["inliers"] >= 0, "invalid C registration inliers")
        if entry.get("status") == "fit":
            direct._require(entry.get("reason") == "fit" and entry["inliers"] >= direct.MIN_INLIERS, "invalid C registration success")
            direct._similarity(np.asarray(entry.get("query_canvas_to_bank"), np.float64))
        else:
            direct._require(entry.get("status") == "failed" and entry.get("reason") in ("registration_failed", "empty_overlap")
                            and entry.get("query_canvas_to_bank") is None, "invalid C registration failure")
    return state


def build_protocol_c_registration(context, *, output_path):
    """Fit authorized remaining queries once and retain every ordinary failure."""
    state = _state(context)
    output = direct._path(output_path)
    protected = [Path(state[name]) for name in ("raw_root", "product_root", "original_root")]
    protected.extend(Path(item["binding"]["path"]) for item in state["authorities"])
    direct._require(not output.exists() and not any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected),
                    "new C registration outside protected inputs required")
    entries, banks = {}, {}
    for row in state["rows"]:
        products = state["banks"][row["design_id"]]["products"]
        if row["design_id"] not in banks:
            target, masks = direct._common(products)
            banks[row["design_id"]] = (direct._features(target), masks)
        features, masks = banks[row["design_id"]]
        path = direct._path(Path(state["raw_root"]) / row["path"])
        query_sha = direct._verifier._read_file(path)[0]
        raw = direct._image(dict(path=str(path), sha256=query_sha), cv2.IMREAD_COLOR)
        canvas, valid, unused = direct.to_canvas(raw)
        expected = _entry_binding(state, row, query_sha)
        key, inliers = direct._digest(expected), 0
        try:
            matrix, inliers = direct._fit(direct._features(canvas), features, int(key[:8], 16) & 0x7fffffff)
            direct._require(matrix is not None and type(inliers) is int and inliers >= direct.MIN_INLIERS, "registration failed")
            matrix = direct._similarity(matrix)
            unused, overlap = direct._overlap(valid, masks, matrix)
            reason = "fit" if direct._available(overlap) else "empty_overlap"
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            matrix, reason = None, "registration_failed"
            inliers = inliers if type(inliers) is int and inliers >= 0 else 0
        entries[row["image_id"]] = dict(expected, key=key, status="fit" if reason == "fit" else "failed", reason=reason,
            inliers=inliers, query_canvas_to_bank=matrix.tolist() if reason == "fit" else None)
    document = dict(schema_version=SCHEMA, context_sha256=context.sha256, policy=state["policy"], entries=entries)
    _validate(context, document)
    payload = _encode(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    return RegistrationTable(payload, _sha(payload), str(output))


def load_protocol_c_registration(context, binding):
    payload = direct._bytes(binding)
    document = direct._verifier._json(payload)
    state = _validate(context, document)
    for row in state["rows"]:
        direct._bytes(dict(path=str(Path(state["raw_root"]) / row["path"]), sha256=document["entries"][row["image_id"]]["query_sha256"]))
    return RegistrationTable(payload, _sha(payload), str(direct._path(binding["path"])))


class ProtocolCReferenceDataset(Dataset):
    def __init__(self, context, table, *, reference_enabled):
        direct._require(type(reference_enabled) is bool and isinstance(table, RegistrationTable)
                        and _sha(table.payload) == table.sha256, "boolean arm and authenticated immutable C registration required")
        direct._bytes(dict(path=table.path, sha256=table.sha256))
        state = _validate(context, direct._verifier._json(table.payload))
        self._context, self._table, self.reference_enabled = context, table, reference_enabled
        self._length = len(state["rows"])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        state = _state(self._context)
        direct._require(_sha(self._table.payload) == self._table.sha256, "C registration snapshot changed")
        direct._bytes(dict(path=self._table.path, sha256=self._table.sha256))
        document = direct._verifier._json(self._table.payload)
        row = state["rows"][index]
        entry = document["entries"][row["image_id"]]
        products = state["banks"][row["design_id"]]["products"]
        query_binding = dict(path=str(Path(state["raw_root"]) / row["path"]), sha256=entry["query_sha256"])
        direct._bytes(query_binding)
        for name in ("geometry", "manifest", "target_valid", "valid", "zone"):
            direct._bytes(products[name])
        if self.reference_enabled and entry["status"] == "fit":
            direct._bytes(products["median"])
        unused, masks = direct._common(products, registration=False)
        raw = direct._image(query_binding, cv2.IMREAD_COLOR)
        query, query_valid, unused = direct.to_canvas(raw)
        overlap, zone = np.zeros((512, 800), np.uint8), np.zeros((512, 800), np.uint8)
        reference, reason = None, entry["reason"]
        if entry["status"] == "fit":
            inverse, overlap = direct._overlap(query_valid, masks, np.asarray(entry["query_canvas_to_bank"]))
            zone = (direct._warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap
            if self.reference_enabled and direct._available(overlap):
                median = direct._image(products["median"], cv2.IMREAD_COLOR)
                direct._require(median.shape == (512, 800, 3), "C median must be 512x800 BGR")
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
            label=int(row["label"]), meta=dict(image_id=row["image_id"], design_id=row["design_id"], role="test", protocol="C",
                attack_family=row["attack_family"], bank="enrollment", registration_key=entry["key"],
                registration_table_sha256=self._table.sha256, query_sha256=entry["query_sha256"],
                availability_reason=reason, requires_referral=not available))
        if self.reference_enabled:
            result["reference"] = torch.from_numpy(np.ascontiguousarray(direct._normalize(reference))) if available else torch.zeros((3, 512, 800), dtype=torch.float32)
        return result
