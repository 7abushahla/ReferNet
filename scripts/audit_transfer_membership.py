"""Read CSV metadata only; freeze existing C roles, never assign enrollment roles.

SHA256 identifies input bytes and can optionally enforce caller-provided pins.
It is not dataset authentication by itself. Recorded document identity/source/
donor tokens and precomputed components are checked without guessing lineage
from filenames. The approved reservation policy must be applied by a separately
reviewed selector. This reader never opens images, annotations, or model files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys


RECORD_FIELDS = {"image_id", "dataset", "design_id", "label", "component_id",
                 "identity_id", "source_id", "donor_id", "path"}
SPLIT_FIELDS = {"image_id", "protocol", "role", "subset", "component_id", "design_id", "label"}
ROLES = ("train", "val", "test", "reference_only")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_metadata(path, fields, expected_sha256=None):
    path = Path(path)
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    require(expected_sha256 is None or checksum == expected_sha256, f"SHA256 mismatch: {path}")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames))
            and fields <= set(reader.fieldnames), f"missing/duplicate CSV columns: {path}")
    rows = list(reader)
    require(rows and all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"empty/malformed CSV: {path}")
    require(all(row["image_id"] and row["component_id"] and row["design_id"] and row["label"] in ("0", "1")
                for row in rows), f"invalid membership fields: {path}")
    require(len({row["image_id"] for row in rows}) == len(rows), f"duplicate image IDs: {path}")
    return rows, dict(path=str(path.resolve()), sha256=checksum, n_bytes=len(content), n_rows=len(rows))


def lineage_metadata(records):
    tokens = defaultdict(set)
    for row in records:
        for field in ("identity_id", "source_id", "donor_id"):
            if row[field]:
                tokens[row[field]].add(row["component_id"])
    conflicts = {token: sorted(components) for token, components in sorted(tokens.items()) if len(components) > 1}
    fantasy = [row for row in records if row["dataset"] == "fantasyid"]
    fields = ("identity_id", "source_id", "donor_id", "component_id")
    return dict(recorded_token_components_consistent=not conflicts, token_component_conflicts=conflicts,
                fantasyid=dict(n_records=len(fantasy),
                               populated={field: sum(bool(row[field]) for row in fantasy) for field in fields},
                               distinct_ids={field: len({row[field] for row in fantasy if row[field]}) for field in fields},
                               altered_missing_source_ids=sum(row["label"] == "1" and not row["source_id"] for row in fantasy)),
                complete_donor_provenance_proven=False,
                scope="Recorded document-token consistency only; blank/unrecorded donors and true cross-file lineage are not inferred.")


def audit_membership(records_path, split_path, *, records_sha256=None, split_sha256=None, d_split=None):
    records, records_binding = read_metadata(records_path, RECORD_FIELDS, records_sha256)
    split, split_binding = read_metadata(split_path, SPLIT_FIELDS, split_sha256)
    indexed = {row["image_id"]: row for row in records}
    joined = []
    for row in split:
        require(row["protocol"] == "C" and row["role"] in ROLES, "existing Protocol C roles required")
        require(row["image_id"] in indexed, "split ID absent from records")
        record = indexed[row["image_id"]]
        require(all(row[field] == record[field] for field in ("design_id", "component_id", "label")),
                "records/split membership mismatch")
        require(record["identity_id"], "recorded document identity required")
        if row["role"] in ("train", "val"):
            require(record["dataset"] in ("sidtd", "idnet") and not row["subset"], "unexpected C source role")
        else:
            require(record["dataset"] == "fantasyid", "C target must be FantasyID")
            if row["role"] == "reference_only":
                require(row["label"] == "0" and not row["subset"], "declared support must be genuine")
            else:
                require(row["subset"] in ("attack1", "heldout"), "unknown C target subset")
        joined.append(dict(record, **{field: row[field] for field in ("protocol", "role", "subset")}))
    joined.sort(key=lambda row: row["image_id"])
    source = [row for row in joined if row["role"] == "train"]
    validation = [row for row in joined if row["role"] == "val"]
    queries = [row for row in joined if row["role"] == "test"]
    supports = [row for row in joined if row["role"] == "reference_only"]
    require(source and validation and queries, "source train/validation and target membership required")
    support_designs = {row["design_id"] for row in supports}
    query_components = {row["component_id"] for row in queries}
    query_identities = {row["identity_id"] for row in queries}
    exclusions = {}
    for query in queries:
        candidates = [row for row in supports if row["design_id"] == query["design_id"]]
        identities = {row["image_id"] for row in candidates if row["identity_id"] == query["identity_id"]}
        components = {row["image_id"] for row in candidates if row["component_id"] == query["component_id"]}
        exclusions[query["image_id"]] = dict(design_id=query["design_id"], declared_candidates=len(candidates),
                                             identity_matches=len(identities), component_matches=len(components),
                                             excluded_union=len(identities | components),
                                             remaining_declared_candidates=len(candidates) - len(identities | components))
    d_status = "not_inspected" if d_split is None else "present_unreviewed" if Path(d_split).is_file() else "absent"
    missing = {subset: sorted({row["design_id"] for row in queries if row["subset"] == subset} - support_designs)
               for subset in sorted({row["subset"] for row in queries})}
    return dict(schema_version="transfer_membership_readiness_v1", metadata_only=True, pixel_reads=0,
                inputs=dict(records=records_binding, split=split_binding),
                membership={role: sorted(row["image_id"] for row in joined if row["role"] == role) for role in ROLES},
                source=dict(n_train=len(source), n_validation=len(validation),
                            datasets=dict(sorted(Counter(row["dataset"] for row in source).items())),
                            identity_overlap_train_validation=sorted({row["identity_id"] for row in source} & {row["identity_id"] for row in validation}),
                            component_overlap_train_validation=sorted({row["component_id"] for row in source} & {row["component_id"] for row in validation})),
                target=dict(n_queries=len(queries), subsets={subset: dict(n_rows=len(subset_rows),
                            labels=dict(sorted(Counter(row["label"] for row in subset_rows).items())),
                            designs=dict(sorted(Counter(row["design_id"] for row in subset_rows).items())))
                            for subset in sorted(missing) for subset_rows in [[row for row in queries if row["subset"] == subset]]}),
                support=dict(n_declared_rows=len(supports), designs=dict(sorted(Counter(row["design_id"] for row in supports).items())),
                             missing_designs_by_subset=missing,
                             shared_component_ids=sorted({row["component_id"] for row in supports} & query_components),
                             shared_identity_ids=sorted({row["identity_id"] for row in supports} & query_identities),
                             per_query_exclusions=exclusions),
                support_policy="approved_reservation_not_applied_existing_roles_only",
                lineage_metadata=lineage_metadata(records),
                d=dict(split_path=None if d_split is None else str(Path(d_split).absolute()), split_status=d_status,
                       idnet_record_count=sum(row["dataset"] == "idnet" for row in records), ready=False),
                limitations=["No reservation, new role, split, threshold, support product, or target pixel access is performed.",
                             "Existing support counts and candidate exclusions are metadata observations, not adopted enrollment authority.",
                             "Input byte hashes freeze this snapshot; without external pins they are not independent authentication.",
                             "Dataset availability on another host and product/runner readiness are not established."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--records-sha256")
    parser.add_argument("--split-sha256")
    parser.add_argument("--d-split", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_membership(args.records, args.split, records_sha256=args.records_sha256,
                                  split_sha256=args.split_sha256, d_split=args.d_split)
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps(dict(schema_version="transfer_membership_readiness_v1", status="error", error=str(error))), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
