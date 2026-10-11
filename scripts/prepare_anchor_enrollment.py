"""Build new fold-safe support products from explicit records and split paths."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.data.enrollment import build_enrollment_products, make_enrollment_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--specimen-root", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output already exists; choose a new versioned directory: {args.out}")
    records_bytes, split_bytes = args.records.read_bytes(), args.split.read_bytes()
    records = pd.read_csv(io.BytesIO(records_bytes), keep_default_na=False)
    split = pd.read_csv(io.BytesIO(split_bytes), keep_default_na=False)
    manifest = make_enrollment_manifest(records, split, args.fold, args.raw_root)
    manifest["input_files"] = {
        "records": {"path": str(args.records.resolve()), "sha256": hashlib.sha256(records_bytes).hexdigest()},
        "split": {"path": str(args.split.resolve()), "sha256": hashlib.sha256(split_bytes).hexdigest()},
        "preparation_script": {"path": str(Path(__file__).resolve()),
                               "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    result = build_enrollment_products(manifest, args.raw_root, args.out, args.specimen_root)
    print(json.dumps({"schema_version": result["schema_version"], "fold": result["fold"],
                      "output_root": result["output_root"], "counts": result["counts"]}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
