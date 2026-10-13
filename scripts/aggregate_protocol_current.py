"""Aggregate explicit, authenticated A/B paired TEST metadata, never run models.

protocol_current_aggregate_request_v1 has protocol (A/B), expected_indices
(exactly 0..9), and units. Each unit has index, split, report_request, and
scorer_result, using absolute path/SHA256 bindings. The report request uses
current_matched_report_request_v1. Completed current A fold-zero or generic
A/B evaluation results are accepted; no artifact directories are discovered.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import report_current_matched as reporter
from refer_or_clear.protocol_a_contract import RECORDS_SHA256, load_protocol_a_contract
from refer_or_clear.protocol_b_contract import load_protocol_b_contract, reserve_target_enrollment

require = reporter.require


class Metadata:
    def __init__(self):
        self.consumed = {}

    def read(self, binding, suffix):
        reporter.validate_binding(binding)
        path = Path(binding["path"])
        require(path.suffix.lower() == suffix and ".." not in path.parts
                and not any(parent.is_symlink() for parent in (path, *path.parents)),
                "regular explicit JSON/CSV metadata path required")
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        require(checksum == binding["sha256"], "metadata SHA256 mismatch")
        self.include({str(path): checksum})
        return reporter.json_object(content) if suffix == ".json" else reporter.csv_rows(content)

    def include(self, hashes):
        for path, checksum in hashes.items():
            require(path not in self.consumed or self.consumed[path] == checksum, "conflicting metadata bindings")
            self.consumed[path] = checksum

    def recheck(self):
        for path, checksum in self.consumed.items():
            self.read(dict(path=path, sha256=checksum), Path(path).suffix.lower())


def _memberships(metadata, request, expected):
    for role, identifiers in expected.items():
        rows = metadata.read(request["memberships"][role], ".csv")
        indexed = reporter.unique_rows(rows, role)
        require(set(indexed) == set(identifiers) and all(row.get("role") == role for row in rows),
                "exact official scored val/test membership required")


def _parent(metadata, unit, request, report, protocol, index, query_ids):
    result = metadata.read(unit["scorer_result"], ".json")
    if result.get("schema_version") == "protocol_a_current_score_result_v1":
        require(protocol == "A" and index == 0 and request.get("scorer_result") == unit["scorer_result"]
                and report.get("scorer_result") == unit["scorer_result"], "legacy A scorer parent binding mismatch")
        return
    key = "fold" if protocol == "A" else "group"
    require(result.get("schema_version") == f"protocol_{protocol.lower()}_evaluation_result_v1"
            and result.get("protocol") == protocol and type(result.get(key)) is int and result[key] == index
            and result.get("passed") is True and result.get("same_query_population") is True,
            "completed current evaluation result required")
    require("scorer_result" not in request, "generic evaluator uses unchanged inline summary provenance")
    require(result.get("summaries") == {arm: request["arms"][arm]["test"]["summary"] for arm in ("off", "on")},
            "evaluation result summary bindings mismatch")
    source_request = metadata.read(result["request"], ".json")
    require(source_request.get("schema_version") == f"protocol_{protocol.lower()}_request_v1"
            and source_request.get("protocol") == protocol and type(source_request.get(key)) is int
            and source_request[key] == index and source_request.get("records") == request["records"]
            and source_request.get("split") == unit["split"]
            and source_request.get("source_sha256") == report["identities"]["source_sha256"],
            "evaluation request source/membership mismatch")
    authorization = metadata.read(result["authorization"], ".json")
    require(authorization.get("schema_version") == f"protocol_{protocol.lower()}_evaluation_v1"
            and authorization.get("protocol") == protocol and type(authorization.get(key)) is int
            and authorization[key] == index and authorization.get("source_validation_selected") is True
            and authorization.get("authorized_image_ids") == query_ids
            and authorization.get("checkpoints") == {arm: report["arms"][arm]["checkpoint"] for arm in ("off", "on")},
            "evaluation authorization population/checkpoint mismatch")
    for name, expected in (("records_sha256", report["identities"]["records_sha256"]),
                           ("split_sha256", report["identities"]["split_sha256"]),
                           ("enrollment_manifest_sha256", report["identities"]["enrollment_sha256"])):
        require(authorization.get(name) == expected, "evaluation authorization identity mismatch")
    for arm in ("off", "on"):
        summary = metadata.read(request["arms"][arm]["test"]["summary"], ".json")
        if "authorization" in summary:
            require(summary["authorization"] == result["authorization"], "summary authorization mismatch")


def _summary(values):
    defined = [value for value in values if value is not None]
    return dict(mean=statistics.mean(defined) if defined else None,
                sd=statistics.stdev(defined) if len(defined) > 1 else None,
                n_defined=len(defined), n_undefined=len(values) - len(defined))


def build_aggregate(request_binding):
    metadata = Metadata()
    request = metadata.read(request_binding, ".json")
    require(set(request) == {"schema_version", "protocol", "expected_indices", "units"}
            and request["schema_version"] == "protocol_current_aggregate_request_v1"
            and request["protocol"] in ("A", "B"), "explicit A/B aggregate request required")
    require(request["expected_indices"] == list(range(10))
            and all(type(index) is int for index in request["expected_indices"]), "exact ten expected indices required")
    require(isinstance(request["units"], list), "explicit units list required")
    indices = []
    for unit in request["units"]:
        require(isinstance(unit, dict) and set(unit) == {"index", "split", "report_request", "scorer_result"}
                and type(unit["index"]) is int and 0 <= unit["index"] <= 9, "explicit fold/group and bound unit artifacts required")
        indices.append(unit["index"])
    require(len(indices) == len(set(indices)), "duplicate fold/group index")
    protocol, per_unit, seen, pooled = request["protocol"], [], Counter(), {arm: [] for arm in ("off", "on")}
    populations, official_ids = {}, None
    for unit in sorted(request["units"], key=lambda value: value["index"]):
        index = unit["index"]
        report_request = metadata.read(unit["report_request"], ".json")
        require(report_request.get("protocol") == protocol and type(report_request.get("index")) is int
                and report_request["index"] == index, "report fold/group/protocol mismatch")
        records_binding = report_request["records"]
        require(records_binding["sha256"] == RECORDS_SHA256, "original reviewed records required")
        metadata.read(records_binding, ".csv")
        metadata.read(unit["split"], ".csv")
        if protocol == "A":
            contract = load_protocol_a_contract(records_binding["path"], unit["split"]["path"],
                fold=index, records_sha256=RECORDS_SHA256)
            query_ids = [row.image_id for row in contract.target_test]
            current_official = {row.image_id for row in contract.source_train + contract.source_val + contract.target_test}
            require(official_ids is None or official_ids == current_official, "A official population differs across folds")
            official_ids = current_official
            require(not set(query_ids) & set(seen), "duplicate A test query across folds")
        else:
            contract = load_protocol_b_contract(records_binding["path"], unit["split"]["path"],
                group=index, records_sha256=RECORDS_SHA256)
            query_ids = reserve_target_enrollment(contract)["query_ids"]
        _memberships(metadata, report_request, dict(val=[row.image_id for row in contract.source_val], test=query_ids))
        key = "fold" if protocol == "A" else "group"
        for arm in ("off", "on"):
            completion = metadata.read(report_request["arms"][arm]["completion"], ".json")
            require("protocol" not in completion or completion["protocol"] == protocol, "completion protocol mismatch")
            require((key not in completion or type(completion[key]) is int and completion[key] == index)
                    and ("group" if protocol == "A" else "fold") not in completion, "completion fold/group index mismatch")
        report = reporter.build_report(unit["report_request"], bootstrap=False)
        metadata.include(report["metadata_sha256"])
        require(report["identities"]["split_sha256"] == unit["split"]["sha256"], "training split identity mismatch")
        _parent(metadata, unit, report_request, report, protocol, index, query_ids)
        paired = {}
        row = dict(index=index, n_test=len(query_ids), report_request=unit["report_request"],
                   scorer_result=unit["scorer_result"], split=unit["split"], identities=report["identities"], arms={})
        for arm in ("off", "on"):
            saved = reporter.unique_rows(metadata.read(report_request["arms"][arm]["test"]["predictions"], ".csv"), arm)
            require(set(saved) == set(query_ids), "matched official test query population required")
            paired[arm] = [dict(saved[identifier], label=reporter.binary_label(saved[identifier]["label"]),
                bonafide_score=reporter.finite_probability(saved[identifier]["bonafide_score"])) for identifier in query_ids]
            native = report["arms"][arm]["test"]["native"]
            require(reporter.native_metrics(paired[arm]) == native, "test metadata changed after report validation")
            row["arms"][arm] = dict(accuracy=native["accuracy"], auc=native["auc"], selected_epoch=report["arms"][arm]["selected_epoch"])
            pooled[arm].extend(paired[arm])
        for first, second in zip(paired["off"], paired["on"]):
            require(all(first[field] == second[field] for field in ("image_id", "label", "design_id", "attack_family")),
                    "OFF/ON test metadata mismatch")
        seen.update(query_ids)
        populations[index] = set(query_ids)
        per_unit.append(row)
    completed = sorted(indices)
    aggregate = {arm: {metric: _summary([row["arms"][arm][metric] for row in per_unit])
                      for metric in ("accuracy", "auc")} for arm in ("off", "on")}
    pooled_result = dict(status="not_applicable_overlapping_b_groups" if protocol == "B" else "unavailable_incomplete_coverage")
    if protocol == "A" and len(completed) == 10:
        require(len(seen) == 2222 and set(seen) == official_ids and all(count == 1 for count in seen.values()),
                "full A OOF requires the exact 2222-query partition")
        pooled_result = dict(status="complete", n_images=len(seen), arms={})
        for arm in ("off", "on"):
            native = reporter.native_metrics(pooled[arm])
            pooled_result["arms"][arm] = {metric: native[metric] for metric in ("accuracy", "auc")}
    overlap = dict(n_scored_observations=sum(seen.values()), n_unique_queries=len(seen),
                  n_repeated_query_ids=sum(count > 1 for count in seen.values()),
                  max_unit_appearances=max(seen.values(), default=0),
                  pairwise=[dict(first=first, second=second, n_common_queries=len(populations[first] & populations[second]))
                            for position, first in enumerate(completed) for second in completed[position + 1:]])
    metadata.recheck()
    return dict(schema_version="protocol_current_aggregate_v1", protocol=protocol, request=request_binding,
        authorization_granted=False, completed_coverage=dict(completed=len(completed), expected=10, fraction=len(completed) / 10),
        completed_indices=completed, missing_indices=sorted(set(range(10)) - set(completed)), per_unit=per_unit,
        aggregate=aggregate, pooled_oof=pooled_result, query_overlap=overlap, metadata_sha256=metadata.consumed,
        score_policy="TEST only; saved bona fide scores; fixed 0.5 threshold; ties genuine; AUC genuine-positive",
        sd_policy="unweighted sample SD (ddof=1) across completed units, undefined with fewer than two defined values; not a confidence interval",
        aggregation_interpretation=("Official A fold averages; incomplete coverage is not the completed ten-fold anchor. "
            "Pooled OOF uses fold-specific selected models and is not the mean fold AUC." if protocol == "A" else
            "Conditional target-group averages with overlapping designs/queries, not independent replicates or an independent CI; no pooled OOF."),
        limitations="Metadata-only descriptive reporting, no pixels, checkpoints, learning, test tuning, model selection, bootstrap, or launch authority. "
            "Upstream checked scorers retain source/checkpoint-byte, completion/history, actual GPU gate, and fresh-process replay proofs.")


def render_table(result):
    lines = [f"Completed {result['completed_coverage']['completed']}/10; missing indices: {result['missing_indices']}",
             "| Index | TEST N | OFF accuracy | ON accuracy | OFF AUC | ON AUC |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in result["per_unit"]:
        lines.append(f"| {row['index']} | {row['n_test']} | {row['arms']['off']['accuracy']} | {row['arms']['on']['accuracy']} | "
                     f"{row['arms']['off']['auc']} | {row['arms']['on']['auc']} |")
    lines.extend(["", "| Arm | TEST accuracy mean | SD | Defined units | TEST AUC mean | SD | Defined units |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for arm, values in result["aggregate"].items():
        accuracy, auc = values["accuracy"], values["auc"]
        lines.append(f"| {arm} | {accuracy['mean']} | {accuracy['sd']} | {accuracy['n_defined']} | "
                     f"{auc['mean']} | {auc['sd']} | {auc['n_defined']} |")
    lines.extend([result["score_policy"], result["aggregation_interpretation"], result["sd_policy"]])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--format", choices=("json", "table"), default="json")
    args = parser.parse_args(argv)
    try:
        result = build_aggregate(dict(path=str(Path(args.request).absolute()), sha256=args.request_sha256))
        content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out is not None:
            require(args.out.suffix.lower() == ".json", "new JSON output required")
            with args.out.open("x") as stream:
                stream.write(content)
        print(content if args.format == "json" else render_table(result), end="")
    except (ValueError, OSError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, f"protocol aggregate failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
