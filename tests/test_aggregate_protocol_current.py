import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.protocol_a_contract import load_protocol_a_contract, RECORDS_SHA256
from refer_or_clear.protocol_b_contract import load_protocol_b_contract, reserve_target_enrollment
from test_report_current_matched import binding, csv_file, document, reporter


@pytest.fixture
def api():
    path = ROOT / "scripts/aggregate_protocol_current.py"
    assert path.is_file(), "current protocol aggregate missing"
    spec = importlib.util.spec_from_file_location("aggregate_protocol_current_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def catalogue(tmp_path, reporter):
    def make(protocol="A", indices=(0, 4, 9), legacy=False):
        records_binding = binding(ROOT / "data/processed/records.csv")
        records = {row["image_id"]: row for row in csv.DictReader(Path(records_binding["path"]).open())}
        units = []
        for index in indices:
            directory = tmp_path / f"{protocol}{index}"
            directory.mkdir()
            split = binding(ROOT / f"data/processed/splits/protocol_{protocol.lower()}_{'it' if protocol == 'A' else 'g'}{index}.csv")
            if protocol == "A":
                contract = load_protocol_a_contract(records_binding["path"], split["path"], fold=index, records_sha256=RECORDS_SHA256)
                query_ids = [row.image_id for row in contract.target_test]
            else:
                contract = load_protocol_b_contract(records_binding["path"], split["path"], group=index, records_sha256=RECORDS_SHA256)
                query_ids = reserve_target_enrollment(contract)["query_ids"]
            role_ids = dict(val=[row.image_id for row in contract.source_val], test=query_ids)
            request = dict(schema_version="current_matched_report_request_v1", protocol=protocol, index=index,
                records=records_binding, memberships={}, arms={})
            for role, identifiers in role_ids.items():
                request["memberships"][role] = csv_file(directory / f"{role}-membership.csv",
                    [dict(image_id=identifier, role=role) for identifier in identifiers])
            key = "fold" if protocol == "A" else "group"
            checkpoints, train_identities = {}, {}
            for arm in ("off", "on"):
                checkpoint = dict(path=str(directory / f"{arm}-checkpoint-NEVER-OPEN.pt"), sha256=("a" if arm == "off" else "b") * 64)
                checkpoints[arm] = checkpoint
                roles = {}
                for role, identifiers in role_ids.items():
                    predictions = []
                    for position, identifier in enumerate(identifiers):
                        record = records[identifier]
                        label = int(record["label"])
                        score = .8 if label == 0 else .2
                        if role == "val": score = 1 - score
                        elif arm == "off" and position % (13 + index) == 0: score = .5
                        predictions.append(dict(image_id=identifier, label=label, design_id=record["design_id"],
                            attack_family=record["attack_family"], bonafide_score=score, attack_probability=1 - score,
                            context_available=True, referral_required=False))
                    predicted = csv_file(directory / f"{arm}-{role}.csv", predictions)
                    native = reporter.native_metrics(predictions)
                    summary = dict(role=role, selected_epoch=10 + index, n_images=len(predictions),
                        accuracy=native["accuracy"], auc=native["auc"], protocol=protocol, **{key: index})
                    if role == "test": summary.update(source_checkpoint=checkpoint, predictions=predicted)
                    roles[role] = dict(predictions=predicted, summary=document(directory / f"{arm}-{role}.json", summary))
                identity = dict(source_sha256="c" * 64, records_sha256=records_binding["sha256"],
                    split_sha256=split["sha256"], enrollment_sha256="e" * 64, registration_sha256="f" * 64)
                train_identities[arm] = identity
                receipt = dict(passed=True, diagnostic_only=False, phase="train", arm=arm, protocol=protocol,
                    epochs_completed=100, selected_epoch=10 + index, replay=dict(passed=True),
                    checkpoint=checkpoint, predictions=roles["val"]["predictions"], identities=identity, **{key: index})
                request["arms"][arm] = dict(completion=document(directory / f"{arm}-train.json", receipt), **roles)
            auth = document(directory / "authorization.json", dict(schema_version=f"protocol_{protocol.lower()}_evaluation_v1",
                protocol=protocol, source_validation_selected=True, records_sha256=records_binding["sha256"],
                split_sha256=split["sha256"], enrollment_manifest_sha256="e" * 64,
                authorized_image_ids=query_ids, checkpoints=checkpoints, **{key: index}))
            parent_request = document(directory / "evaluation-request.json", dict(schema_version=f"protocol_{protocol.lower()}_request_v1",
                protocol=protocol, source_sha256="c" * 64, records=records_binding, split=split, **{key: index}))
            if legacy:
                assert protocol == "A" and index == 0
                parent_arms = {}
                for arm, inputs in request["arms"].items():
                    path = Path(inputs["test"]["summary"]["path"])
                    summary = json.loads(path.read_text())
                    summary.pop("source_checkpoint")
                    summary.pop("predictions")
                    summary.update(aggregate=False, selected_on="source_validation")
                    inputs["test"]["summary"] = document(path, summary)
                    parent_arms[arm] = dict(checkpoint=checkpoints[arm], train_result=inputs["completion"],
                        training_identities=train_identities[arm], **inputs["test"])
                parent = dict(schema_version="protocol_a_current_score_result_v1", passed=True, protocol="A", fold=0,
                    role="test", n_images=len(query_ids), image_ids=query_ids, aggregate=False,
                    seed=42, eval_batch_size=1, learning_performed=False, threshold_tuning_performed=False,
                    training_source_sha256="c" * 64, arms=parent_arms)
                scorer = document(directory / "result.json", parent)
                request["scorer_result"] = scorer
            else:
                parent = dict(schema_version=f"protocol_{protocol.lower()}_evaluation_result_v1", passed=True,
                    protocol=protocol, same_query_population=True, request=parent_request, authorization=auth,
                    summaries={arm: inputs["test"]["summary"] for arm, inputs in request["arms"].items()}, **{key: index})
                scorer = document(directory / "result.json", parent)
            units.append(dict(index=index, split=split, report_request=document(directory / "report-request.json", request),
                              scorer_result=scorer))
        request = dict(schema_version="protocol_current_aggregate_request_v1", protocol=protocol,
            expected_indices=list(range(10)), units=units)
        return document(tmp_path / f"aggregate-{protocol}.json", request)
    return make


def update(path_binding, change):
    path = Path(path_binding["path"])
    value = json.loads(path.read_text())
    change(value)
    return document(path, value)


def test_partial_a_native_test_only_missing_explicit_and_sample_sd(api, catalogue, reporter):
    authority = catalogue()
    result = api.build_aggregate(authority)
    assert result["completed_coverage"] == dict(completed=3, expected=10, fraction=.3)
    assert result["completed_indices"] == [0, 4, 9]
    assert result["missing_indices"] == [1, 2, 3, 5, 6, 7, 8]
    assert [row["n_test"] for row in result["per_unit"]] == [222, 223, 223]
    assert result["aggregate"]["on"]["accuracy"] == dict(mean=1.0, sd=0.0, n_defined=3, n_undefined=0)
    assert result["pooled_oof"]["status"] == "unavailable_incomplete_coverage"
    reports = [reporter.build_report(unit["report_request"]) for unit in json.loads(Path(authority["path"]).read_text())["units"]]
    values = [report["arms"]["off"]["test"]["native"]["accuracy"] for report in reports]
    import statistics
    assert result["aggregate"]["off"]["accuracy"]["mean"] == pytest.approx(statistics.mean(values))
    assert result["aggregate"]["off"]["accuracy"]["sd"] == pytest.approx(statistics.stdev(values))
    assert result["authorization_granted"] is False


def test_full_a_pools_exact_2222_unique_queries_only(api, catalogue):
    result = api.build_aggregate(catalogue(indices=tuple(range(10))))
    assert result["completed_coverage"]["completed"] == 10
    assert result["missing_indices"] == []
    assert result["pooled_oof"]["status"] == "complete"
    assert result["pooled_oof"]["n_images"] == 2222
    assert result["pooled_oof"]["arms"]["on"] == dict(accuracy=1.0, auc=1.0)
    assert result["query_overlap"]["n_repeated_query_ids"] == 0


def test_b_group_overlap_is_conditional_not_pooled_or_ci(api, catalogue):
    result = api.build_aggregate(catalogue("B", (0, 1, 2)))
    assert result["query_overlap"]["n_repeated_query_ids"] > 0
    assert result["query_overlap"]["n_scored_observations"] > result["query_overlap"]["n_unique_queries"]
    assert result["pooled_oof"]["status"] == "not_applicable_overlapping_b_groups"
    assert "conditional" in result["aggregation_interpretation"].lower() and "independent" in result["aggregation_interpretation"]
    assert "confidence_interval" not in json.dumps(result)


def test_legacy_actual_a_parent_plain_summary_compatible(api, catalogue):
    result = api.build_aggregate(catalogue(indices=(0,), legacy=True))
    assert result["completed_indices"] == [0]
    assert result["aggregate"]["on"]["accuracy"]["sd"] is None


def test_zero_coverage_is_not_fabricated(api, tmp_path):
    authority = document(tmp_path / "empty.json", dict(schema_version="protocol_current_aggregate_request_v1",
        protocol="A", expected_indices=list(range(10)), units=[]))
    result = api.build_aggregate(authority)
    assert result["completed_coverage"]["completed"] == 0
    assert result["aggregate"]["off"]["auc"]["mean"] is None
    assert result["missing_indices"] == list(range(10))


def test_undefined_descriptive_stats_are_explicit(api):
    assert api._summary([None, .5, .7]) == dict(mean=.6, sd=pytest.approx(2**.5 / 10), n_defined=2, n_undefined=1)
    assert api._summary([None]) == dict(mean=None, sd=None, n_defined=0, n_undefined=1)


def test_a_crossfold_duplicate_guard_is_independent_of_count(api, catalogue, monkeypatch):
    authority = catalogue(indices=(0, 4))
    original = api.load_protocol_a_contract
    first = None
    def duplicated(*args, **kwargs):
        nonlocal first
        actual = original(*args, **kwargs)
        if first is None:
            first = actual
            return actual
        from dataclasses import replace
        return replace(actual, source_train=first.source_train, source_val=first.source_val, target_test=first.target_test)
    monkeypatch.setattr(api, "load_protocol_a_contract", duplicated)
    with pytest.raises(ValueError, match="duplicate A test query"):
        api.build_aggregate(authority)


def test_completion_declared_fold_cannot_disagree_with_bound_index(api, catalogue):
    authority = catalogue(indices=(4,))
    value = json.loads(Path(authority["path"]).read_text())
    unit = value["units"][0]
    report = json.loads(Path(unit["report_request"]["path"]).read_text())
    report["arms"]["off"]["completion"] = update(report["arms"]["off"]["completion"], lambda receipt: receipt.update(fold=9))
    unit["report_request"] = document(Path(unit["report_request"]["path"]), report)
    authority = document(Path(authority["path"]), value)
    with pytest.raises(ValueError, match="completion.*index"):
        api.build_aggregate(authority)


@pytest.mark.parametrize("change", ["index", "bool-index", "duplicate-unit", "expected", "protocol", "split", "scorer-missing", "scorer-hash", "report-hash"])
def test_invalid_catalogue_fails_closed(api, catalogue, change):
    authority = catalogue(indices=(4,))
    def corrupt(value):
        unit = value["units"][0]
        if change == "index": unit["index"] = 9
        elif change == "bool-index": unit["index"] = True
        elif change == "duplicate-unit": value["units"].append(copy.deepcopy(unit))
        elif change == "expected": value["expected_indices"] = [4]
        elif change == "protocol": value["protocol"] = "C"
        elif change == "split": unit["split"] = binding(ROOT / "data/processed/splits/protocol_a_it9.csv")
        elif change == "scorer-missing": unit.pop("scorer_result")
        elif change == "scorer-hash": unit["scorer_result"]["sha256"] = "0" * 64
        else: unit["report_request"]["sha256"] = "0" * 64
    with pytest.raises(ValueError): api.build_aggregate(update(authority, corrupt))


@pytest.mark.parametrize("change", ["index", "passed", "summaries", "authorization", "request-source", "off-on-population", "duplicate-query", "test-is-val", "incomplete-training", "wrong-summary"])
def test_hash_consistent_broken_parent_or_chain_rejected(api, catalogue, change):
    authority = catalogue(indices=(4,))
    value = json.loads(Path(authority["path"]).read_text())
    unit = value["units"][0]
    report = json.loads(Path(unit["report_request"]["path"]).read_text())
    scorer = json.loads(Path(unit["scorer_result"]["path"]).read_text())
    if change in ("index", "passed", "summaries"):
        if change == "index": scorer["fold"] = 9
        elif change == "passed": scorer["passed"] = False
        else: scorer["summaries"]["off"] = scorer["summaries"]["on"]
    elif change == "authorization":
        scorer["authorization"] = update(scorer["authorization"], lambda auth: auth.update(authorized_image_ids=[]))
    elif change == "request-source":
        scorer["request"] = update(scorer["request"], lambda request: request.update(source_sha256="0" * 64))
    elif change == "incomplete-training":
        report["arms"]["off"]["completion"] = update(report["arms"]["off"]["completion"], lambda receipt: receipt.update(epochs_completed=99))
    elif change == "wrong-summary":
        report["arms"]["off"]["test"]["summary"] = update(report["arms"]["off"]["test"]["summary"], lambda summary: summary.update(accuracy=0.0))
    elif change == "test-is-val": report["memberships"]["test"] = report["memberships"]["val"]
    else:
        predicted = report["arms"]["on"]["test"]["predictions"]
        rows = list(csv.DictReader(Path(predicted["path"]).open()))
        if change == "duplicate-query": rows[1] = dict(rows[0])
        else: rows.pop()
        report["arms"]["on"]["test"]["predictions"] = csv_file(Path(predicted["path"]), rows)
    unit["scorer_result"] = document(Path(unit["scorer_result"]["path"]), scorer)
    unit["report_request"] = document(Path(unit["report_request"]["path"]), report)
    authority = document(Path(authority["path"]), value)
    with pytest.raises(ValueError): api.build_aggregate(authority)


def test_no_pixels_or_checkpoints_read_and_mutation_detection(api, catalogue, monkeypatch):
    authority = catalogue(indices=(0,))
    original = Path.read_bytes
    def metadata_only(path):
        assert path.suffix in (".json", ".csv"), f"nonmetadata read: {path}"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    api.build_aggregate(authority)
    original_report = api.reporter.build_report
    def racing(binding_value, **kwargs):
        result = original_report(binding_value, **kwargs)
        path = Path(binding_value["path"])
        path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(api.reporter, "build_report", racing)
    with pytest.raises(ValueError, match="changed|SHA256|conflict"): api.build_aggregate(authority)


def test_import_has_no_models_pixels_or_torch():
    script = ROOT / "scripts/aggregate_protocol_current.py"
    code = f"""
import builtins, importlib.util
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert not name.startswith(('torch','cv2','PIL','refer_or_clear.models','refer_or_clear.data.direct_reference','protocol_a_study','protocol_b_study')), name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('aggregate_import_check', {str(script)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_hash_binding_and_exclusive_output(api, catalogue, tmp_path):
    authority = catalogue(indices=(9,))
    output = tmp_path / "aggregate.json"
    assert api.main(["--request", authority["path"], "--request-sha256", authority["sha256"], "--out", str(output)]) == 0
    first = output.read_bytes()
    with pytest.raises(SystemExit): api.main(["--request", authority["path"], "--request-sha256", authority["sha256"], "--out", str(output)])
    assert output.read_bytes() == first
    with pytest.raises(SystemExit): api.main(["--request", authority["path"], "--request-sha256", "0" * 64])


def test_table_displays_test_descriptive_means_and_sd(api, catalogue):
    result = api.build_aggregate(catalogue(indices=(0, 4)))
    table = api.render_table(result)
    assert "Completed 2/10" in table
    assert "TEST accuracy mean" in table and "TEST AUC mean" in table and "SD" in table
    assert "unweighted" in table and "ties genuine" in table
