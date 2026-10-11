#!/usr/bin/env python3
"""Prepare recipient-only SIDTD development masks without changing source data.

The output is an audited mask overlay, not a complete training dataset. It omits
test examples and does not change image labels, folds, canvases or references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.data.forgeries import build_masks
from refer_or_clear.data.records import sidtd_target_field


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents or output.exists():
        raise ValueError("output must be a new directory outside the source data tree")


def development_records(records, split):
    selected = split[split.role.isin(("train", "val"))][["image_id", "label", "role"]]
    joined = selected.merge(records, on="image_id", how="left", validate="one_to_one", suffixes=("_split", ""))
    if joined.dataset.isna().any() or not (joined.label == joined.label_split).all():
        raise ValueError("missing records or mismatched split labels")
    if not (joined.dataset == "sidtd").all():
        raise ValueError("only SIDTD development images are supported")
    return joined.drop(columns="label_split")


def pixel_audit(raw_root, record, annotation):
    source_path = raw_root / "sidtd/templates/Images/reals" / annotation["src"]
    fake = cv2.imread(str(raw_root / record.path), cv2.IMREAD_COLOR)
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None or fake is None or source.shape != fake.shape:
        raise ValueError(f"invalid source/forgery geometry for {record.image_id}")
    design, number = Path(annotation["src"]).stem.rsplit("_", 1)
    via = json.loads((raw_root / f"sidtd/templates/Annotations/reals/{design}.json").read_text())
    entry = next(value for value in via["_via_img_metadata"].values() if value["filename"] == number + ".jpg")
    rectangles = {region["region_attributes"]["field_name"]: region["shape_attributes"]
                  for region in entry["regions"] if region["shape_attributes"]["name"] == "rect"}
    difference = np.abs(cv2.GaussianBlur(source.astype(np.float32), (0, 0), 1)
                        - cv2.GaussianBlur(fake.astype(np.float32), (0, 0), 1)).mean(2)
    result = dict(raw_source_sha256=digest(source_path), raw_forgery_sha256=digest(raw_root / record.path))
    for key in ("field", "second_field"):
        rectangle = rectangles[annotation[key]]
        left, top = rectangle["x"], rectangle["y"]
        patch = difference[top:top + rectangle["height"], left:left + rectangle["width"]]
        result[f"{key}_mean_absolute_difference"] = float(patch.mean())
        result[f"{key}_fraction_above_10"] = float((patch > 10).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/processed")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source, raw_root, output, report = map(Path, (args.source, args.raw_root, args.out, args.report))
    validate_output(source, output)
    validate_output(source, report)
    if output.resolve() == report.resolve() or output.resolve() in report.resolve().parents or report.resolve() in output.resolve().parents:
        raise ValueError("mask output and audit report require separate directories")
    manifests = [source / filename for filename in ("records.csv", "transforms.json", "splits/protocol_a_it0.csv")]
    original_hashes = {str(path): digest(path) for path in manifests}
    records = pd.read_csv(manifests[0]).fillna("")
    split = pd.read_csv(manifests[2]).fillna("")
    selected = development_records(records, split)
    assert selected.role.value_counts().to_dict() == {"train": 1778, "val": 222}
    forged = selected[selected.label == 1].copy()
    annotations = {}
    for record in forged.itertuples(index=False):
        annotation = json.loads((raw_root / record.annotation).read_text())
        annotations[record.image_id] = annotation
        selected.loc[selected.image_id == record.image_id, "altered_fields"] = sidtd_target_field(annotation)
        path = source / "masks" / f"{record.image_id}.png"
        original_hashes[str(path)] = digest(path)
    transforms = json.loads(manifests[1].read_text())
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    build_masks(selected, raw_root, transforms, output / "masks")
    selected.to_csv(output / "records_train_val.csv", index=False)
    rows = []
    for record in forged.itertuples(index=False):
        old_path = source / "masks" / f"{record.image_id}.png"
        new_path = output / "masks" / f"{record.image_id}.png"
        old_mask = cv2.imread(str(old_path), cv2.IMREAD_GRAYSCALE)
        new_mask = cv2.imread(str(new_path), cv2.IMREAD_GRAYSCALE)
        assert old_mask.shape == new_mask.shape == (512, 800)
        assert np.any(new_mask) and not np.any((new_mask > 0) & (old_mask == 0))
        annotation = annotations[record.image_id]
        changed = not np.array_equal(old_mask, new_mask)
        different_fields = record.attack_family == "text_copy" and annotation["field"] != annotation["second_field"]
        assert changed == different_fields, record.image_id
        row = dict(image_id=record.image_id, role=record.role, family=record.attack_family,
                   recipient_field=annotation["field"], donor_field=annotation.get("second_field", ""),
                   changed=changed, old_pixels=int((old_mask > 0).sum()), new_pixels=int((new_mask > 0).sum()),
                   old_sha256=original_hashes[str(old_path)], new_sha256=digest(new_path))
        if different_fields:
            row.update(pixel_audit(raw_root, record, annotation))
        rows.append(row)
    assert all(digest(path) == expected for path, expected in original_hashes.items())
    table = pd.DataFrame(rows)
    table.to_csv(report / "mask_comparison.csv", index=False)
    changed = table[table.changed]
    summary = dict(complete=True, source=str(source), output=str(output), standalone_dataset=False,
                   roles=selected.role.value_counts().to_dict(), test_masks_prepared=0,
                   forged_masks=len(rows), changed_masks=len(changed),
                   changes_by_role=changed.role.value_counts().to_dict(), source_unchanged=True,
                   source_hashes=original_hashes,
                   code_hashes={str(path.relative_to(ROOT)): digest(path) for path in
                                (Path(__file__).resolve(), ROOT / "src/refer_or_clear/data/records.py",
                                 ROOT / "src/refer_or_clear/data/forgeries.py")},
                   interpretation="Recipient-only box targets; no measured training benefit yet.")
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith("hashes")}, indent=2))


if __name__ == "__main__":
    main()
