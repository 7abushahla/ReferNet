"""Freeze reviewed B membership and build only authorized genuine support."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.protocol_b_contract import (
    load_protocol_b_contract, load_target_enrollment, reserve_target_enrollment,
)
from refer_or_clear.data.protocol_b_enrollment import (
    make_protocol_b_enrollment_manifest, build_protocol_b_enrollment_products,
)


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
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def prepare(records, records_sha256, split, group, raw_root, output_root, *, build):
    contract = load_protocol_b_contract(records, split, group=group, records_sha256=records_sha256)
    output = Path(output_root).absolute()
    if output.exists():
        raise FileExistsError(f"preparation already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    target_binding = write(output / "target-enrollment.json", reserve_target_enrollment(contract))
    target = load_target_enrollment(contract, Path(target_binding["path"]), expected_sha256=target_binding["sha256"])
    manifest = make_protocol_b_enrollment_manifest(Path(records), contract, target, Path(raw_root))
    authorized_binding = write(output / "authorized-enrollment.json", manifest)
    products_binding = None
    if build:
        build_protocol_b_enrollment_products(manifest, Path(raw_root), output / "enrollment")
        products_path = output / "enrollment/manifest.json"
        products_binding = dict(path=str(products_path.resolve()), sha256=digest(products_path))
    result = dict(schema_version="protocol_b_preparation_v1", passed=True, protocol="B", group=group,
                  phase="products" if build else "metadata", records_sha256=contract.records_sha256,
                  split_sha256=contract.split_sha256, target_enrollment=target_binding,
                  authorized_enrollment=authorized_binding, enrollment=products_binding,
                  n_train=len(contract.source_train), n_val=len(contract.source_val),
                  n_target_support=len(target.support_ids), n_target_queries=len(target.query_ids),
                  n_target_excluded=len(target.excluded_ids), target_query_pixels_opened=False)
    write(output / "preparation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "records-sha256", "split", "raw-root", "output-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--group", type=int, choices=range(10), required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(Path(args.records), args.records_sha256, Path(args.split), args.group,
                     Path(args.raw_root), Path(args.output_root), build=args.build)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
