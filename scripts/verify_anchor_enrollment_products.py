"""Verify completed fold-zero bank products, not detector or model clearance.

Only declared bank output files are opened. PNGs are streamed for SHA256 without
decoding. Geometry and dependency JSON are checked against the authenticated
completed manifest. --output-root optionally relocates the exact declared root;
it never authorizes paths outside that original root or another bank directory.
The report must be a new file outside both output trees and the manifest folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

SCHEMA_VERSION = "anchor_enrollment_v1"
GEOMETRY_VERSION = "bank_similarity_v1"
BANKS = {"train_half0", "train_half1", "enrollment"}
REQUIRED_PRODUCTS = {"target", "target_coverage", "target_valid", "median", "coverage", "valid", "zone", "geometry", "manifest"}
OPTIONAL_PRODUCTS = {"specimen", "specimen_valid"}
CANVAS_HW = [512, 800]
AXES = "x=column,y=row; native_to_bank=canvas_to_bank@native_to_canvas"


def _sha256_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _absolute(value):
    if not isinstance(value, (str, Path)) or not str(value) or "\\" in str(value):
        raise ValueError("invalid absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute path without parent traversal required")
    return path


def _no_symlinks(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"symlink path forbidden: {parent}")


def _read_file(path, capture=False):
    _no_symlinks(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("regular output file required")
        checksum = hashlib.sha256()
        content = []
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
            if capture:
                content.append(chunk)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("file changed while reading")
    return checksum.hexdigest(), b"".join(content) if capture else None


def _json(content):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    value = json.loads(content, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _error(result, code, message, **context):
    result["errors"].append({"code": code, "message": str(message), **context})


def _finish(result):
    result["error_count"] = len(result["errors"])
    result["passed"] = not result["errors"]
    return result


def _matrix(value):
    return (isinstance(value, list) and len(value) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in value)
            and all(type(number) in (int, float) and math.isfinite(number) for row in value for number in row))


def _identifiers(value):
    return (isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)
            and len(value) == len(set(value)))


def _check_bank_json(result, manifest, selected, documents, context):
    geometry = documents.get("geometry")
    dependency = documents.get("manifest")
    if geometry is None or dependency is None:
        _error(result, "bank_json_unavailable", "geometry/dependency JSON pair unavailable", **context)
    if dependency is not None:
        expected = {name: entry.get("sha256") for name, entry in selected["products"].items()
                    if name != "manifest" and isinstance(entry, dict)}
        if dependency.get("output_sha256") != expected:
            _error(result, "dependency_output_sha256", "dependency outputs must equal product table excluding manifest self-hash", **context)
    if geometry is None:
        return
    expected_fields = {"schema_version": SCHEMA_VERSION, "geometry_version": GEOMETRY_VERSION, "fold": 0,
                       "design_id": context["design"], "bank": context["bank"], "canvas_hw": CANVAS_HW, "axes": AXES,
                       "valid_interpolation": "nearest", "rgb_interpolation": "linear",
                       "support_ids": selected.get("support_ids"), "registered_ids": selected.get("registered_ids"),
                       "contributors": selected.get("contributors"), "source": manifest.get("source"),
                       "official_split_sha256": manifest.get("official_split_sha256")}
    for field, expected in expected_fields.items():
        if field not in geometry or geometry[field] != expected or (field == "fold" and type(geometry[field]) is not int):
            _error(result, "geometry_binding", f"geometry {field} differs from completed manifest/geometry contract", **context)
    support = selected.get("support_ids")
    registered = selected.get("registered_ids")
    if _identifiers(support) and geometry.get("anchor_id") != support[0]:
        _error(result, "geometry_anchor", "anchor must be the first ordered authorized support", **context)
    registrations = geometry.get("registrations")
    if not isinstance(registrations, dict) or not _identifiers(support) or set(registrations) != set(support):
        _error(result, "geometry_registrations", "registration keys differ from support membership", **context)
    elif _identifiers(registered):
        for identifier, registration in registrations.items():
            if not isinstance(registration, dict):
                _error(result, "geometry_registration", f"invalid registration for {identifier}", **context)
                continue
            wanted = identifier in registered
            if type(registration.get("target_applied")) is not bool or registration["target_applied"] != wanted:
                _error(result, "geometry_registration", f"target application differs from registered membership for {identifier}", **context)
            for field in ("canvas_to_bank", "native_to_bank"):
                if (_matrix(registration.get(field)) if wanted else registration.get(field) is None) is not True:
                    _error(result, "geometry_registration", f"invalid {field} for {identifier}", **context)
    if dependency is not None:
        result["counts"]["bank_json_pairs_checked"] += 1
        if {key: value for key, value in dependency.items() if key != "output_sha256"} != geometry:
            _error(result, "dependency_geometry", "dependency JSON must equal geometry plus output_sha256 only", **context)
    has_specimen = "specimen" in selected["products"]
    if has_specimen != ("specimen" in geometry):
        _error(result, "geometry_specimen", "specimen geometry/product presence differs", **context)


def verify_products(manifest_path, expected_manifest_sha256, *, output_root=None):
    manifest_path = Path(manifest_path).absolute()
    result = {"schema_version": "anchor_enrollment_integrity_v1", "scope": "bank_product_integrity_only",
              "manifest_path": str(manifest_path), "expected_manifest_sha256": expected_manifest_sha256,
              "actual_manifest_sha256": None, "declared_output_root": None, "verification_output_root": None,
              "detector_clearance": False, "pixels_decoded": False, "raw_or_query_files_opened": False,
              "errors": [], "products": [], "counts": {"designs_declared": 0, "banks_checked": 0,
              "products_declared": 0, "products_hashed": 0, "products_matched": 0, "products_missing": 0,
              "bank_json_pairs_checked": 0, "registered_supports": 0}}
    try:
        actual, content = _read_file(manifest_path, capture=True)
        result["actual_manifest_sha256"] = actual
        if not _sha256_value(expected_manifest_sha256) or actual != expected_manifest_sha256:
            try:
                result["declared_output_root"] = str(_absolute(_json(content).get("output_root")))
            except (ValueError, UnicodeError):
                pass
            _error(result, "manifest_sha256", "manifest differs from external SHA256 authority")
            return _finish(result)
        manifest = _json(content)
    except (OSError, ValueError, UnicodeError) as error:
        _error(result, "manifest_read", error)
        return _finish(result)
    for field, wanted in (("schema_version", SCHEMA_VERSION), ("geometry_version", GEOMETRY_VERSION), ("fold", 0),
                          ("official_fold_binding", {"protocol": "A", "fold": 0, "ref_group": "it0"}),
                          ("classifier_counts", {"train": 1778, "val": 222, "test": 222}),
                          ("official_split_hash_format", "canonical_json_rows_v1")):
        if manifest.get(field) != wanted or (field == "fold" and type(manifest.get(field)) is not int):
            _error(result, "manifest_contract", f"unsupported or inconsistent {field}")
    binding = manifest.get("official_fold_binding")
    if (not _sha256_value(manifest.get("official_split_sha256")) or not isinstance(binding, dict)
            or type(binding.get("fold")) is not int):
        _error(result, "manifest_contract", "official split/fold authority is malformed")
    if not _identifiers(manifest.get("eligible_ids")):
        _error(result, "manifest_membership", "nonempty unique eligible support IDs required")
    banks = manifest.get("banks")
    if not isinstance(banks, dict) or not banks:
        _error(result, "manifest_banks", "completed manifest requires nonempty named banks")
    try:
        declared = _absolute(manifest.get("output_root"))
        actual_root = _absolute(output_root) if output_root is not None else declared
        result.update(declared_output_root=str(declared), verification_output_root=str(actual_root))
        _no_symlinks(actual_root)
    except (OSError, ValueError) as error:
        _error(result, "output_root", error)
    if result["errors"]:
        return _finish(result)
    result["counts"]["designs_declared"] = len(banks)
    paths_seen = set()
    for design, named in sorted(banks.items()):
        design_path = Path(design)
        if (design_path.is_absolute() or "\\" in design or any(part in ("", ".", "..") for part in design.split("/"))
                or not isinstance(named, dict)):
            _error(result, "bank_design", "invalid design path or bank table", design=design)
            continue
        if set(named) != BANKS:
            _error(result, "bank_names", "every design requires train_half0, train_half1 and enrollment", design=design)
        for bank, selected in sorted(named.items()):
            context = {"design": design, "bank": bank}
            result["counts"]["banks_checked"] += 1
            if bank not in BANKS or not isinstance(selected, dict):
                _error(result, "bank_contract", "invalid named bank", **context)
                continue
            for field, wanted in (("design_id", design), ("bank", bank), ("fold", 0), ("geometry_version", GEOMETRY_VERSION)):
                if selected.get(field) != wanted or (field == "fold" and type(selected.get(field)) is not int):
                    _error(result, "bank_contract", f"bank {field} mismatch", **context)
            support, registered = selected.get("support_ids"), selected.get("registered_ids")
            if (not _identifiers(support) or not _identifiers(registered) or not set(registered).issubset(support)
                    or not isinstance(selected.get("contributors"), dict) or set(selected["contributors"]) != set(support)):
                _error(result, "bank_membership", "support, registration or contributor membership inconsistent", **context)
            else:
                result["counts"]["registered_supports"] += len(registered)
                if selected.get("counts") != {"authorized": len(support), "registered": len(registered), "failed": len(support) - len(registered)}:
                    _error(result, "bank_counts", "bank counts differ from membership", **context)
            products = selected.get("products")
            if not isinstance(products, dict) or not products:
                _error(result, "bank_products", "bank requires nonempty product table", **context)
                continue
            if not REQUIRED_PRODUCTS.issubset(products) or set(products) - REQUIRED_PRODUCTS - OPTIONAL_PRODUCTS:
                _error(result, "bank_products", "missing required or unknown product names", **context)
            if bool(set(products) & OPTIONAL_PRODUCTS) and not OPTIONAL_PRODUCTS.issubset(products):
                _error(result, "bank_products", "specimen products must occur as a pair", **context)
            documents = {}
            for product, entry in sorted(products.items()):
                result["counts"]["products_declared"] += 1
                record = {**context, "product": product, "declared_path": entry.get("path") if isinstance(entry, dict) else None,
                          "verified_path": None, "expected_sha256": entry.get("sha256") if isinstance(entry, dict) else None,
                          "actual_sha256": None, "matched": False}
                result["products"].append(record)
                product_context = {**context, "product": product, "path": record["declared_path"]}
                try:
                    if product not in REQUIRED_PRODUCTS | OPTIONAL_PRODUCTS or not isinstance(entry, dict):
                        raise ValueError("unsupported product declaration")
                    original_path = _absolute(entry.get("path"))
                    expected_path = declared / "fold0" / design_path / bank / f"{product}{'.json' if product in ('geometry', 'manifest') else '.png'}"
                    if not original_path.is_relative_to(declared) or original_path != expected_path:
                        raise ValueError("product must remain in its exact authorized bank/output-root path")
                    mapped = actual_root / original_path.relative_to(declared)
                    _no_symlinks(mapped)
                    if mapped in paths_seen:
                        raise ValueError("duplicate product file path")
                    paths_seen.add(mapped)
                    record["verified_path"] = str(mapped)
                except (OSError, ValueError) as error:
                    _error(result, "product_path", error, **product_context)
                    continue
                if not _sha256_value(entry.get("sha256")):
                    _error(result, "product_authority", "malformed declared SHA256", **product_context)
                    continue
                try:
                    observed, content = _read_file(mapped, capture=product in ("geometry", "manifest"))
                    record["actual_sha256"] = observed
                    result["counts"]["products_hashed"] += 1
                    record["matched"] = observed == entry["sha256"]
                    if record["matched"]:
                        result["counts"]["products_matched"] += 1
                    else:
                        _error(result, "product_sha256", "actual product SHA256 differs from declared authority", **product_context)
                    if content is not None:
                        try:
                            documents[product] = _json(content)
                        except (ValueError, UnicodeError) as error:
                            _error(result, "product_json", error, **product_context)
                except FileNotFoundError as error:
                    result["counts"]["products_missing"] += 1
                    _error(result, "product_missing", error, **product_context)
                except (OSError, ValueError) as error:
                    _error(result, "product_read", error, **product_context)
            _check_bank_json(result, manifest, selected, documents, context)
    expected_counts = {"designs": len(banks), "banks": sum(len(named) for named in banks.values() if isinstance(named, dict)),
                       "eligible": len(manifest.get("eligible_ids", []))}
    if manifest.get("counts") != expected_counts:
        _error(result, "manifest_counts", "completed manifest counts differ from declared membership")
    return _finish(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, help="explicit local relocation of the exact declared output root")
    args = parser.parse_args(argv)
    report = args.report.absolute()
    _no_symlinks(report)
    report = report.resolve()
    if (report.exists() or report.is_relative_to(args.manifest.absolute().parent)
            or report.is_relative_to(args.manifest.absolute().parent.resolve())):
        raise ValueError("report must be new and outside the manifest/output tree")
    if args.output_root is not None:
        root = _absolute(args.output_root)
        if report.is_relative_to(root) or report.is_relative_to(root.resolve()):
            raise ValueError("report must be outside the relocated output tree")
    result = verify_products(args.manifest, args.manifest_sha256, output_root=args.output_root)
    for field in ("declared_output_root", "verification_output_root"):
        if result[field] is not None:
            root = Path(result[field])
            if report.is_relative_to(root) or report.is_relative_to(root.resolve()):
                raise ValueError("report must be outside original and relocated output trees")
    report.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(report)
    with report.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
