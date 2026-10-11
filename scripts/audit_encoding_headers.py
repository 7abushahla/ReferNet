"""Audit frozen FantasyID membership and encoding headers without decoding pixels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import struct
from collections import Counter
from pathlib import Path

from PIL import Image


class UnsupportedEncodingError(ValueError):
    def __init__(self, payload_format):
        super().__init__(f"unsupported payload format {payload_format}")
        self.payload_format = payload_format


STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
STD_CHROMA = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _signature(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _ijg_table(base, quality):
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return [min(255, max(1, (entry * scale + 50) // 100)) for entry in base]


def read_encoding_header(path: Path) -> dict:
    """Inspect JPEG/PNG headers only; validity of the pixel payload is not asserted."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(33)
        stream.seek(0)
        with Image.open(stream) as opened:
            payload_format = opened.format
            if payload_format not in {"JPEG", "MPO", "PNG"}:
                raise UnsupportedEncodingError(payload_format)
            tables = {str(key): list(value) for key, value in
                      getattr(opened, "quantization", {}).items()}
            components = [dict(component_id=component[0], horizontal=component[1],
                               vertical=component[2], table_id=component[3])
                          for component in getattr(opened, "layer", [])]
            markers = {"mode": opened.mode}
            if payload_format in {"JPEG", "MPO"}:
                markers.update({key: opened.info[key] for key in
                                ("jfif", "jfif_version", "adobe", "adobe_transform", "progressive")
                                if key in opened.info})
            else:
                if len(prefix) < 33 or prefix[:8] != b"\x89PNG\r\n\x1a\n" or prefix[12:16] != b"IHDR":
                    raise ValueError("invalid PNG IHDR")
                markers.update(dict(zip(
                    ("bit_depth", "color_type", "compression", "filter", "interlace"),
                    struct.unpack("BBBBB", prefix[24:29]),
                )))
            candidates = []
            if tables and set(tables).issubset({"0", "1"}):
                for quality in range(1, 101):
                    standard = {"0": _ijg_table(STD_LUMA, quality),
                                "1": _ijg_table(STD_CHROMA, quality)}
                    if all(table == standard[key] for key, table in tables.items()):
                        candidates.append(quality)
            estimate = None
            if "0" in tables and len(tables["0"]) == 64:
                scale = statistics.median(100 * value / base for value, base in zip(tables["0"], STD_LUMA))
                if scale > 0:
                    estimate = round(5000 / scale if scale > 100 else (200 - scale) / 2, 1)
            return dict(
                format=payload_format, width=opened.width, height=opened.height,
                color_markers=markers, components=components, quantization_tables=tables,
                table_signature=_signature(tables) if tables else None,
                sampling_signature=_signature([[component["horizontal"], component["vertical"]]
                                               for component in components]) if components else None,
                exact_ijg_candidates=candidates,
                table_status=("standard_ijg_match" if candidates else
                              "custom_or_unmatched" if tables else "not_applicable"),
                estimated_quality=estimate, estimate_method="legacy_luminance_median_ratio",
            )


def _read_csv(payload, required, name):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"{name} lacks required columns {sorted(required)}")
    rows = list(reader)
    identifiers = [row["image_id"] for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} image_id must be nonempty and unique")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{name} contains malformed CSV rows")
    return rows


def _counts(rows):
    statuses = Counter(row["status"] for row in rows)
    return dict(total=len(rows), genuine=sum(row["label"] == 0 for row in rows),
                altered=sum(row["label"] == 1 for row in rows),
                **{status: statuses[status] for status in
                   ("ok", "missing", "corrupt", "unsupported", "access_error", "error")})


def _metrics(rows):
    eligible = [row for row in rows if row.get("estimated_quality") is not None]
    genuine = sorted(-row["estimated_quality"] for row in eligible if row["label"] == 0)
    altered = sorted(-row["estimated_quality"] for row in eligible if row["label"] == 1)
    auc = None
    if genuine and altered:
        from bisect import bisect_left, bisect_right

        wins = sum(bisect_left(genuine, score) +
                   (bisect_right(genuine, score) - bisect_left(genuine, score)) / 2
                   for score in altered)
        auc = wins / (len(genuine) * len(altered))
    return dict(eligible=len(eligible), excluded=len(rows) - len(eligible),
                genuine=len(genuine), altered=len(altered),
                rule="estimated_quality < 85", auc_direction="-estimated_quality",
                accuracy_estimate_lt_85=(sum((row["estimated_quality"] < 85) == row["label"]
                                            for row in eligible) / len(eligible) if eligible else None),
                auc_negative_estimate=auc)


def _signature_association(rows, field):
    eligible = [row for row in rows if row["status"] == "ok" and row.get(field)]
    counts = Counter((row[field], row["label"]) for row in eligible)
    genuine = {row[field] for row in eligible if row["label"] == 0}
    altered = {row[field] for row in eligible if row["label"] == 1}
    disjoint = genuine.isdisjoint(altered) if genuine and altered else None
    full_coverage = len(eligible) == len(rows) and bool(rows)
    return dict(eligible=len(eligible), excluded=len(rows) - len(eligible),
                eligible_class_counts=dict(genuine=sum(row["label"] == 0 for row in eligible),
                                           altered=sum(row["label"] == 1 for row in eligible)),
                coverage=len(eligible) / len(rows) if rows else None, full_coverage=full_coverage,
                class_disjoint=disjoint, population_class_disjoint=disjoint if full_coverage else None,
                signatures=[dict(signature=signature, genuine=counts[signature, 0], altered=counts[signature, 1])
                            for signature in sorted(genuine | altered)])


def _group(rows):
    strata = Counter((row.get("design_id", ""), row.get("attack_family", ""),
                      row.get("capture", ""), row["label"], row["status"],
                      row.get("format", ""), row.get("table_signature"),
                      row.get("sampling_signature")) for row in rows)
    fields = ("design_id", "attack_family", "capture", "label", "status",
              "format", "table_signature", "sampling_signature")
    return dict(counts=_counts(rows), metrics=_metrics(rows),
                signature_associations={"quantization": _signature_association(rows, "table_signature"),
                                        "sampling": _signature_association(rows, "sampling_signature"),
                                        "format": _signature_association(rows, "format")},
                strata=[dict(zip(fields, key), count=count)
                        for key, count in sorted(strata.items(), key=lambda item: str(item[0]))])


def _support_report(records, audited):
    parent = {}

    def find(token):
        parent.setdefault(token, token)
        while parent[token] != token:
            parent[token] = parent[parent[token]]
            token = parent[token]
        return token

    def tokens(row):
        dataset = row["dataset"].lower()
        values = {("identity", dataset, row[field]) for field in ("identity_id", "source_id", "donor_id")
                  if row.get(field)}
        if row.get("component_id"):
            values.add(("component", dataset, row["component_id"]))
        return values

    for row in records:
        members = sorted(tokens(row))
        for token in members:
            parent[find(token)] = find(members[0])
    evaluation = [row for row in audited if row["role"] == "test"]
    support = [row for row in audited if row["role"] == "reference_only"]
    evaluation_tokens = set().union(*(tokens(row) for row in evaluation))
    support_tokens = set().union(*(tokens(row) for row in support))
    evaluation_components = {find(token) for token in evaluation_tokens}
    overlapping = [row["image_id"] for row in support
                   if any(find(token) in evaluation_components for token in tokens(row))]
    unknown = [row["image_id"] for row in support if not row.get("identity_id")]
    unknown_evaluation = [row["image_id"] for row in evaluation if not row.get("identity_id")]
    uncertain_designs = {row["design_id"] for row in evaluation if row["image_id"] in unknown_evaluation}
    usable_designs = {row["design_id"] for row in support if row["label"] == 0
                      and row["status"] == "ok" and row["image_id"] not in overlapping + unknown
                      and row["design_id"] not in uncertain_designs}
    return dict(approved=False, inventory_only=True,
                provenance_semantics="source_id and donor_id are identity IDs, not image IDs",
                identity_overlap_tokens=sorted({row.get("identity_id") for row in evaluation
                                                if row.get("identity_id")} &
                                               {row.get("identity_id") for row in support
                                                if row.get("identity_id")}),
                provenance_overlap_tokens=sorted(value if namespace == "identity" else "component:" + value
                                                 for namespace, dataset, value in evaluation_tokens & support_tokens),
                provenance_overlap_nodes=[dict(namespace=namespace, dataset=dataset, value=value)
                                          for namespace, dataset, value in sorted(evaluation_tokens & support_tokens)],
                overlapping_support_image_ids=sorted(overlapping),
                unknown_identity_support_image_ids=sorted(unknown),
                unknown_identity_evaluation_image_ids=sorted(unknown_evaluation),
                separate_support_image_ids=sorted(row["image_id"] for row in support
                                                 if row["label"] == 0 and row["status"] == "ok"
                                                 and row["image_id"] not in overlapping + unknown
                                                 and row["design_id"] not in uncertain_designs),
                nongenuine_support_image_ids=sorted(row["image_id"] for row in support if row["label"] != 0),
                designs_without_separate_support=sorted({row["design_id"] for row in evaluation} - usable_designs),
                missing_provenance_columns=sorted({"identity_id", "source_id", "donor_id", "component_id"} -
                                                  (set(records[0]) if records else set())))


def _metadata_quality(records, selected):
    fields = ("identity_id", "source_id", "donor_id", "component_id", "design_id", "attack_family", "capture")
    populated = {field: [row.get(field, "") for row in selected if row.get(field, "").strip()]
                 for field in fields}
    missing_values = {}
    for field in fields:
        if field == "donor_id":
            continue
        required = [row for row in selected if field not in {"source_id", "attack_family"} or row["label"] == 1]
        missing_values[field] = sorted(row["image_id"] for row in required if not row.get(field, "").strip())
    return dict(missing_columns=sorted(set(fields) - (set(records[0]) if records else set())),
                missing_values=missing_values,
                populated_counts={field: len(values) for field, values in populated.items()},
                distinct_counts={field: len(set(values)) for field, values in populated.items()},
                counts_scope="Selected candidate test and reference-only membership; nonblank values only.",
                semantics="Blank source_id and attack_family are expected for genuine rows; donor_id is optional.")


def _disk_inventory(raw_root, official_rows, requested):
    extensions = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".mpo", ".gif", ".bmp",
                  ".tif", ".tiff", ".webp", ".avif", ".heic", ".heif"}
    report = dict(requested=requested, scanned=False, suffix_filter=sorted(extensions),
                  disk_image_count=None, disk_images_absent_records=[], records_absent_disk=[], errors=[])
    if not requested:
        return report
    directory = raw_root / "fantasyid/test"
    disk_paths = set()
    try:
        if not directory.is_dir():
            raise FileNotFoundError(f"test inventory directory unavailable: {directory}")
        for folder, subdirectories, filenames in os.walk(directory, onerror=lambda error: report["errors"].append(str(error))):
            for filename in filenames:
                path = Path(folder) / filename
                if path.suffix.lower() in extensions:
                    disk_paths.add(path.relative_to(raw_root).as_posix())
        report["scanned"] = not report["errors"]
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    record_paths = {Path(row["path"]).as_posix() for row in official_rows}
    report.update(disk_image_count=len(disk_paths),
                  disk_images_absent_records=sorted(disk_paths - record_paths),
                  records_absent_disk=sorted(record_paths - disk_paths))
    return report


def _header_failure(error):
    fields = dict(error=f"{type(error).__name__}: {error}")
    if isinstance(error, FileNotFoundError):
        fields["status"] = "missing"
    elif isinstance(error, UnsupportedEncodingError):
        fields.update(status="unsupported", format=error.payload_format)
    elif isinstance(error, (ValueError, SyntaxError, struct.error, Image.DecompressionBombError)):
        fields["status"] = "corrupt"
    elif isinstance(error, OSError):
        fields["status"] = "access_error" if error.errno is not None else "corrupt"
    else:
        fields["status"] = "error"
    return fields


def audit_target_headers(records_csv: Path, split_csv: Path, raw_root: Path, output_dir: Path,
                         *, inventory_test_directory: bool = False) -> dict:
    """Freeze metadata, join exactly once, and inspect only candidate test/support headers."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"new output directory required: {output_dir}")
    record_bytes, split_bytes = Path(records_csv).read_bytes(), Path(split_csv).read_bytes()
    records = _read_csv(record_bytes, {"image_id", "dataset", "path", "label", "design_id"}, "records")
    splits = _read_csv(split_bytes, {"image_id", "role", "subset"}, "split")
    indexed = {row["image_id"]: row for row in records}
    selected = []
    for split in splits:
        if split["image_id"] not in indexed:
            raise ValueError(f"unknown split image_id {split['image_id']}")
        record = indexed[split["image_id"]]
        for field in (set(split) & set(record)) - {"image_id"}:
            if split[field] != record[field]:
                raise ValueError(f"conflicting {field} for {split['image_id']}")
        if record["dataset"].lower() == "fantasyid" and split["role"] in {"test", "reference_only"}:
            if record["label"] not in {"0", "1"}:
                raise ValueError(f"invalid binary label for {record['image_id']}")
            selected.append(dict(record, role=split["role"], subset=split["subset"], label=int(record["label"])))
    selected.sort(key=lambda row: row["image_id"])
    raw_root = Path(raw_root).resolve()
    resolved_paths = {}
    resolution_failures = {}
    for row in selected:
        try:
            path = (raw_root / row["path"]).resolve()
        except Exception as error:
            resolution_failures[row["image_id"]] = _header_failure(error)
            continue
        if not path.is_relative_to(raw_root) or Path(row["path"]).is_absolute():
            raise ValueError(f"path outside raw root: {row['path']}")
        resolved_paths.setdefault(str(path), []).append(row["image_id"])
    duplicate_paths = [dict(resolved_path=path, image_ids=identifiers)
                       for path, identifiers in sorted(resolved_paths.items()) if len(identifiers) > 1]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "records.csv").write_bytes(record_bytes)
    (output_dir / "protocol_c.csv").write_bytes(split_bytes)
    (output_dir / "membership.json").write_text(json.dumps(selected, indent=2) + "\n")
    for row in selected:
        if row["image_id"] in resolution_failures:
            row.update(resolution_failures[row["image_id"]])
            continue
        try:
            row.update(read_encoding_header(raw_root / row["path"]))
            row["status"], row["error"] = "ok", ""
        except Exception as error:
            row.update(_header_failure(error))
    evaluation = [row for row in selected if row["role"] == "test"]
    support = [row for row in selected if row["role"] == "reference_only"]
    official_rows = [row for row in records if row["dataset"].lower() == "fantasyid"
                     and row["path"].startswith("fantasyid/test/")]
    official_ids = {row["image_id"] for row in official_rows}
    candidate_ids = {row["image_id"] for row in evaluation}
    groups = {"candidate_test": _group(evaluation), "reference_only": _group(support)}
    for subset in sorted({"heldout", "attack1"} | {row["subset"] for row in evaluation}):
        groups[subset or "unspecified_subset"] = _group([row for row in evaluation if row["subset"] == subset])
    inventory = _disk_inventory(raw_root, official_rows, inventory_test_directory)
    metadata = _metadata_quality(records, selected)
    headers_complete = bool(selected) and all(row["status"] == "ok" for row in selected)
    manifest_complete = bool(official_ids) and official_ids == candidate_ids
    inventory_confirmed = (inventory["scanned"] and not inventory["disk_images_absent_records"]
                           and not inventory["records_absent_disk"] and manifest_complete)
    reasons = []
    if not headers_complete:
        reasons.append("Candidate headers are empty or contain missing, corrupt, unsupported, or failed files.")
    if not manifest_complete:
        reasons.append("Candidate test IDs do not exactly match the official-test paths in frozen records.")
    if not inventory_confirmed:
        reasons.append("Exhaustive test membership is not confirmed by the optional suffix-based disk inventory.")
    if duplicate_paths:
        reasons.append("Selected image IDs share resolved paths.")
    if metadata["missing_columns"] or any(metadata["missing_values"].values()):
        reasons.append("Provenance or strata metadata has missing columns or required values.")
    summary = dict(
        schema_version=2,
        membership_version="membership_candidate_v1", raw_root=str(raw_root),
        manifests={"records": {"source": str(Path(records_csv).resolve()),
                                "sha256": hashlib.sha256(record_bytes).hexdigest()},
                   "split": {"source": str(Path(split_csv).resolve()),
                              "sha256": hashlib.sha256(split_bytes).hexdigest()},
                   "membership_sha256": _signature(json.loads((output_dir / "membership.json").read_text()))},
        counts=_counts(selected), groups=groups, support=_support_report(records, selected),
        completeness=dict(complete=not reasons, candidate_headers_complete=headers_complete,
                          manifest_membership_complete=manifest_complete,
                          exhaustive_test_inventory_confirmed=bool(inventory_confirmed), reasons=reasons,
                          scope="Candidate headers, declared metadata, and suffix-filtered test-directory inventory only"),
        disk_inventory=inventory, duplicate_selected_paths=duplicate_paths, metadata_quality=metadata,
        official_test_membership=dict(record_count=len(official_ids),
                                      missing_from_candidate=sorted(official_ids - candidate_ids),
                                      candidate_not_official_path=sorted(candidate_ids - official_ids)),
        omissions=[{"image_id": row["image_id"], "status": row["status"], "error": row["error"],
                    "format": row.get("format")}
                   for row in selected if row["status"] != "ok"],
        limitations=["Header validity does not establish pixel-payload integrity.",
                     "Exact IJG candidates match available full tables, not a recovered encoder setting.",
                     "The luminance-based estimate is approximate, not an exact encoder quality.",
                     "Metrics exclude unavailable estimates and use a fixed threshold and direction.",
                     "Attack1 uses only its existing membership; no genuine comparison pool is assigned.",
                     "Candidate test is the frozen split membership, not an asserted exhaustive official test.",
                     "Optional disk inventory uses filenames/suffixes only, not unselected image headers or release authentication.",
                     "For JPEG-family MPO files, only the primary image header is inspected.",
                     "Signature class-disjointness concerns known eligible headers; excluded rows prevent population-wide conclusions.",
                     "Reference-only is an unapproved inventory; provenance relies on existing metadata.",
                     "Encoding association does not prove exclusive CNN reliance or a causal mechanism.",
                     "A revised evaluation/support contract requires a new membership audit."],
    )
    columns = sorted(set().union(*(set(row) for row in selected))) if selected else ["image_id", "status"]
    with (output_dir / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _json(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in selected)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = ["# membership_candidate_v1", "", "Frozen current membership, not a locked Protocol C loader.", ""]
    if not summary["completeness"]["complete"]:
        report += ["**INCOMPLETE. No exhaustive test-population conclusions are supported.**", ""]
        report += [f"- {reason}" for reason in reasons] + [""]
    report += ["| Group | Total | Genuine | Altered | Missing | Corrupt | Unsupported | Access error | Other error | Estimate eligible | Estimate excluded | Fixed rule accuracy | Fixed direction AUC |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, group in groups.items():
        counts, metrics = group["counts"], group["metrics"]
        report.append("| " + " | ".join(str(value) for value in
                      (name, counts["total"], counts["genuine"], counts["altered"], counts["missing"],
                       counts["corrupt"], counts["unsupported"], counts["access_error"], counts["error"],
                       metrics["eligible"], metrics["excluded"], metrics["accuracy_estimate_lt_85"],
                       metrics["auc_negative_estimate"])) + " |")
    report += ["", "The fixed rule is estimated_quality < 85. AUC uses -estimated_quality.",
               "Single-class or unavailable-estimate AUC is null. Detailed strata are in summary.json.",
               "", "## Signature associations", "",
               "Class-disjointness is conditional on known eligible headers, independent of the fixed estimate threshold or sign.",
               "", "```json", json.dumps({name: group["signature_associations"] for name, group in groups.items()}, indent=2), "```",
               "", "## Candidate support and membership gaps", "", "```json",
               json.dumps(dict(support=summary["support"], official_test_membership=summary["official_test_membership"],
                               completeness=summary["completeness"], disk_inventory=inventory,
                               duplicate_selected_paths=duplicate_paths, metadata_quality=metadata,
                               omissions=summary["omissions"]), indent=2), "```", "", "## Limitations", ""]
    report += [f"- {limitation}" for limitation in summary["limitations"]]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("records", "split", "raw-root", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--inventory-test-directory", action="store_true",
                        help="Compare test-directory image suffix inventory against frozen records without opening unselected files")
    args = parser.parse_args()
    summary = audit_target_headers(args.records, args.split, args.raw_root, args.out,
                                   inventory_test_directory=args.inventory_test_directory)
    print(json.dumps({"schema_version": summary["schema_version"], "membership_version": summary["membership_version"],
                      "counts": summary["counts"], "completeness": summary["completeness"]}))


if __name__ == "__main__":
    main()
