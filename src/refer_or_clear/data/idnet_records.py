"""Metadata-only AZ positive/fraud1-4 ingestion, without assigning dataset roles.

The input is the AZ directory, not a general IDNet archive. Original annotation
keys identify documents. Morph baseface fields provide an explicit bijection
between portrait tokens and those documents; no filename token parsing is used.
Replacement donors must resolve to original documents. Unresolved external donors
and combined morphs without explicit donor provenance are rejected, not inferred
from multi-face filenames. Textfield annotations declare no document donor, so
their empty donor_id is accompanied by not_declared_by_annotation, not a claim
of complete lineage. Basic metadata's is_donor is not a provenance field.

Only annotation JSON bytes and image directory entries are inspected. Width,
height, masks, train/evaluation roles, and splits are deliberately absent. JSON
SHA256 receipts identify the bytes inspected; they do not certify scientific
correctness, release authenticity, or completeness of the publisher's lineage.
Fraud5/6 and WV are outside this parser's scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .records import provenance_components


GROUPS = {
    "positive": ("AZ_original_annotation.json", ""),
    "fraud1_copy_and_move": ("AZ_textfieldreplacement_annotation.json", "text_copy"),
    "fraud2_face_morphing": ("AZ_facemorph_annotation.json", "face_morph"),
    "fraud3_face_replacement": ("AZ_facereplacement_annotation.json", "face_swap"),
    "fraud4_combined": ("AZ_combined_fraud_annotation.json", "mixed"),
}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate annotation key: {key}")
        result[key] = value
    return result


def _filename(value):
    if (not isinstance(value, str) or not value.endswith(".png")
            or "/" in value or "\\" in value or "\x00" in value
            or value in (".png", "..png")):
        raise ValueError(f"invalid annotation filename: {value!r}")
    return value


def _no_symlinks(path):
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"symlink is not permitted: {ancestor}")


def idnet_records(az_root: Path) -> pd.DataFrame:
    """Return metadata records only if all five groups have resolvable lineage."""
    root = Path(az_root).absolute()
    annotations = {}
    hashes = {}
    for group, (filename, _) in GROUPS.items():
        path = root / "meta/detailed_with_fraud_info" / filename
        _no_symlinks(path)
        payload = path.read_bytes()
        metadata = json.loads(payload, object_pairs_hook=_unique_pairs)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(f"annotation must be a nonempty object: {filename}")
        for name, entry in metadata.items():
            _filename(name)
            if not isinstance(entry, dict):
                raise ValueError(f"annotation entry must be an object: {name}")
        directory = root / group
        _no_symlinks(directory)
        inventory = set()
        for image in directory.iterdir():
            _no_symlinks(image)
            if not image.is_file() or image.suffix.lower() != ".png":
                raise ValueError(f"unknown inventory entry: {image.name}")
            inventory.add(image.name)
        if inventory != set(metadata):
            raise ValueError(f"annotation/image inventory mismatch: {group}")
        annotations[group] = metadata
        hashes[filename] = hashlib.sha256(payload).hexdigest()
    originals = set(annotations["positive"])
    if any(set(metadata) != originals for metadata in annotations.values()):
        raise ValueError("all five annotation memberships must match original documents")
    portraits = {}
    for name, entry in annotations["fraud2_face_morphing"].items():
        token = entry.get("baseface")
        if not isinstance(token, str) or not token or token in portraits:
            raise ValueError(f"ambiguous morph baseface binding: {token!r}")
        portraits[token] = name

    def donor_document(token):
        if not isinstance(token, str) or token not in originals:
            raise ValueError(f"unresolved donor document: {token!r}")
        return f"idnet/AZ/{token}"

    rows = []
    for group, (filename, family) in GROUPS.items():
        for name, entry in sorted(annotations[group].items()):
            identity = f"idnet/AZ/{name}"
            donor = ""
            status = "not_applicable" if group == "positive" else "declared"
            if group in ("positive", "fraud1_copy_and_move"):
                expected = "False" if group == "positive" else "True"
                if entry.get("fraud") != expected:
                    raise ValueError(f"annotation label polarity mismatch: {group}/{name}")
                if group != "positive":
                    status = "not_declared_by_annotation"
            elif group == "fraud2_face_morphing":
                token = entry.get("overlayface")
                if not isinstance(token, str) or token not in portraits:
                    raise ValueError(f"unresolved morph donor: {token!r}")
                donor = donor_document(portraits[token])
            else:
                if entry.get("originalface") != name:
                    raise ValueError(f"annotation source mismatch: {group}/{name}")
                if group == "fraud4_combined":
                    flags = (entry.get("face_morphing"), entry.get("face_replacement"))
                    if flags == ("True", "False"):
                        raise ValueError("combined morph has no explicit resolvable donor field")
                    if flags != ("False", "True"):
                        raise ValueError(f"unknown combined face flags: {flags!r}")
                    donor = donor_document(entry.get("replaced_new_face"))
                else:
                    donor = donor_document(entry.get("newface"))
            rows.append(dict(
                dataset="idnet", image_id=f"idnet_AZ_{group}_{name[:-4]}",
                path=f"idnet/AZ/{group}/{name}", design_id="idnet/AZ",
                identity_id=identity, label=int(group != "positive"),
                attack_family=family, attack_source=group, capture="render",
                source_id=identity if group != "positive" else "", donor_id=donor,
                donor_status=status,
                annotation=f"idnet/AZ/meta/detailed_with_fraud_info/{filename}",
                annotation_key=name,
            ))
    result = pd.DataFrame(rows)
    if not result.image_id.is_unique:
        raise ValueError("ambiguous image identifiers")
    result["component_id"] = provenance_components(result)
    result.attrs["annotation_sha256"] = hashes
    result.attrs["metadata_only"] = True
    return result
