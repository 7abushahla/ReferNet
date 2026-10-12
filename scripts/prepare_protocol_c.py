"""Prepare authorized C target support on CPU, never train or select models.

Default metadata mode opens only CSV/reservation bytes. --build explicitly opens
reserved genuine support and its JSON, never target queries or source images.
Use a new output tree outside raw data and input authorities. No A-ON completion
or checkpoint binding is required until the separate evaluation transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.protocol_c_contract import _path, _require, load_protocol_c_contract, load_target_enrollment, reserve_target_enrollment
from refer_or_clear.data.protocol_c_enrollment import make_protocol_c_enrollment_manifest, build_protocol_c_enrollment_products


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(path=str(_path(path)), sha256=digest(path))


def prepare(records, records_sha256, source_split, target_split, raw_root, output_root, *, build):
    _require(type(build) is bool, "build must be boolean")
    contract = load_protocol_c_contract(records, source_split, target_split, records_sha256=records_sha256)
    reservation = reserve_target_enrollment(contract)
    output, raw = _path(output_root), _path(raw_root)
    _require(raw.is_dir() and not output.is_relative_to(raw) and not raw.is_relative_to(output), "output/raw roots must be disjoint")
    for path in (records, source_split, target_split):
        _require(not _path(path).is_relative_to(output), "output overlaps input authority")
    if output.exists():
        raise FileExistsError(f"preparation already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    target_binding = write(output / "target-enrollment.json", reservation)
    target = load_target_enrollment(contract, target_binding["path"], expected_sha256=target_binding["sha256"])
    manifest = make_protocol_c_enrollment_manifest(contract, target, raw)
    authorized_binding = write(output / "authorized-enrollment.json", manifest)
    products_binding = None
    if build:
        build_protocol_c_enrollment_products(manifest, raw, output / "enrollment")
        path = output / "enrollment/manifest.json"
        products_binding = dict(path=str(path), sha256=digest(path))
    result = dict(schema_version="protocol_c_preparation_v1", passed=True, protocol="C", phase="products" if build else "metadata",
        records_sha256=contract.records_sha256, source_split_sha256=contract.source_split_sha256, target_split_sha256=contract.target_split_sha256,
        target_enrollment=target_binding, authorized_enrollment=authorized_binding, enrollment=products_binding,
        n_train=len(contract.source_train), n_val=len(contract.source_val), published_target_count=len(contract.target_test),
        n_target_support=len(target.support_ids), n_target_queries=len(target.query_ids), n_target_excluded=len(target.excluded_ids),
        target_query_pixels_opened=False, source_pixels_opened=False, target_outcomes_measured=False,
        checkpoint_evaluation_authorized=False, lineage=reservation["lineage"])
    write(output / "preparation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "records-sha256", "source-split", "target-split", "raw-root", "output-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.records, args.records_sha256, args.source_split, args.target_split, args.raw_root, args.output_root, build=args.build)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
