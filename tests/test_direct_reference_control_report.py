import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.pilots import compare_replay, json_sha256, summarize_predictions


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return binding(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return binding(path)


@pytest.fixture
def report():
    path = ROOT / "scripts/report_direct_reference_control.py"
    assert path.is_file(), "metadata-only completed-control report is not implemented"
    specification = importlib.util.spec_from_file_location("direct_reference_control_report", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def refresh(fixture):
    predictions = fixture["predictions"]
    summary = summarize_predictions(predictions)
    fixture["train"]["predictions"] = write_csv(fixture["directory"] / "val_predictions.csv", predictions)
    fixture["train"]["selection_key"] = [summary["accuracy"], summary["auc"]]
    history = [dict(epoch=epoch, n_images=1776, optimizer_steps=222, loss=.2, val_image_loss=.2,
                    val_accuracy=summary["accuracy"], val_auc=summary["auc"]) for epoch in range(1, 101)]
    history_binding = write_csv(fixture["directory"] / "log.csv", history)
    fixture["completion"]["train_result"] = write(fixture["directory"] / "train.json", fixture["train"])
    replay = dict(schema_version="direct_reference_control_replay_v1", passed=True, fresh_process=True, phase="train",
                  arm="off", pid=9876, n_images=222, buffers_unchanged=True,
                  train_result=fixture["completion"]["train_result"], control_request=fixture["completion"]["control_request"],
                  identities=fixture["train"]["identities"], checkpoint=fixture["completion"]["checkpoint"],
                  history=history_binding, image_ids=[row["image_id"] for row in predictions], predictions=predictions,
                  summary=dict(summary, buffers_unchanged=True), replay=compare_replay(predictions, predictions), max_abs_difference=0.)
    fixture["completion"]["fresh_replay"] = write(fixture["directory"] / "fresh-replay.json", replay)
    fixture["control_binding"] = write(fixture["directory"] / "control.json", fixture["completion"])
    return fixture


@pytest.fixture
def completed(tmp_path):
    base = tmp_path / "original metadata"
    inputs = base / "staged/inputs"
    directory = base / "run/training/off"
    inputs.mkdir(parents=True)
    directory.mkdir(parents=True)
    config_path = inputs / "config.yaml"
    config_path.write_bytes((ROOT / "configs/direct_reference_pilot.yaml").read_bytes())
    config = yaml.safe_load(config_path.read_text())
    records, split, predictions = [], [], []
    for role, count in (("train", 1778), ("val", 222), ("test", 222)):
        for index in range(count):
            label = int(index >= count // 2)
            family = "text_copy" if role == "val" and index >= 206 else "text_inpaint" if label else ""
            row = dict(image_id=f"{role}{index:04d}", path=f"never-open/{role}/{index}.png", label=label,
                       design_id="design", component_id=f"component{role}{index}", attack_family=family)
            records.append(row)
            split.append({key: row[key] for key in ("image_id", "label", "design_id", "component_id")} | {"role": role})
            if role == "val":
                logit = 2. if label else -2.
                probability = torch.tensor(logit, dtype=torch.float32).sigmoid().item()
                available = index >= 10
                predictions.append({key: row[key] for key in ("image_id", "label", "design_id", "attack_family")} |
                    dict(attack_logit=logit, query_logit=-2., correction_logit=4. if label else 0.,
                         attack_probability=probability, bonafide_score=1. - probability,
                         context_available=available, referral_required=not available))
    authorities = dict(config=binding(config_path), records=write_csv(inputs / "records.csv", records),
                       split=write_csv(inputs / "split.csv", split),
                       weights=dict(path=str(inputs / "weights-must-not-open.pth"), sha256=config["weights_sha256"]))
    request = write(inputs / "request.json", dict(schema_version="direct_reference_diagnostic_request_v1", source_sha256="a" * 64))
    manifest = dict(schema_version="direct_reference_job_v1", diagnostic_only=True, source_sha256="a" * 64,
                    paths=dict(run_dir=str(base / "run")), request=request, authorities=authorities)
    manifest_binding = write(base / "run/input_manifest.json", manifest)
    gate = write(base / "diagnostic.json", dict(schema_version="direct_reference_diagnostic_v1", passed=True,
                  no_test_open=True, diagnostic_only=True, identities=dict(source_sha256="a" * 64)))
    control = write(base / "control-request.json", dict(schema_version="direct_reference_control_request_v1", arm="off",
                    epochs=100, fold=0, seed=42, source_sha256="a" * 64, config_sha256=authorities["config"]["sha256"],
                    diagnostic_request_sha256=request["sha256"], diagnostic_gate=gate))
    identities = dict(source_sha256="a" * 64, config_sha256=json_sha256(config), records_sha256=authorities["records"]["sha256"],
                      split_sha256=authorities["split"]["sha256"], weights_sha256=config["weights_sha256"],
                      enrollment_sha256=config["enrollment_sha256"], registration_sha256="b" * 64,
                      phase_authority_sha256="c" * 64, preprocess_sha256="d" * 64, initialization_sha256="e" * 64)
    checkpoint = dict(path=str(directory / "best.pt"), sha256="f" * 64)
    train = dict(passed=True, phase="train", diagnostic_only=False, epochs_completed=100, selected_epoch=1,
                 optimizer_steps=22200, role="val", arm="off", identities=identities,
                 scheduler=dict(total_updates=22200, warmup_updates=1000), checkpoint=checkpoint,
                 diagnostic_gate=gate, replay=dict(passed=True, fresh_process=False, n_images=222, phase="train"))
    completion = dict(schema_version="direct_reference_control_result_v1", passed=True, phase="train", arm="off",
                      epochs_completed=100, control_request=control, identities=identities, checkpoint=checkpoint)
    return refresh(dict(base=base, directory=directory, manifest_binding=manifest_binding, manifest=manifest,
                        predictions=predictions, train=train, completion=completion))


def test_complete_control_reports_fixed_metrics_and_query_diagnostic(report, completed, monkeypatch):
    def forbidden(*arguments, **keywords):
        raise AssertionError("report must not load weights, images, or construct a model")
    monkeypatch.setattr(torch, "load", forbidden)
    from refer_or_clear import direct_reference_train
    monkeypatch.setattr(direct_reference_train, "_fresh_model", forbidden)
    monkeypatch.setattr(direct_reference_train, "_dataset", forbidden)
    result = report.build_report(completed["control_binding"], completed["manifest_binding"])
    assert result["status"] == "complete" and result["competence"]["passed"] is True
    assert result["combined"]["accuracy"] == result["combined"]["auc"] == 1.
    assert result["combined"]["genuine_false_positive_rate"] == 0.
    assert result["combined"]["subgroups"]["copy"]["n_altered"] == 16
    assert result["context_availability"] == dict(available=212, unavailable=10, total=222)
    assert result["query_component_only"]["accuracy"] == result["query_component_only"]["auc"] == .5
    assert result["decision_changes"]["count"] == 111
    assert result["checkpoint_verification"] == "receipt-binding-only; tensor bytes not opened"
    assert result["family_selection"] == "not-authorized"
    assert "not an independently trained" in result["limitations"]


@pytest.mark.parametrize("mutation", ["root-hash", "manifest-hash", "root-schema", "root-off", "root-passed",
    "train-incomplete", "train-test", "train-on", "identity", "checkpoint", "missing-replay", "replay-passed",
    "replay-order", "replay-identity", "replay-summary", "replay-flag-only", "csv-order", "csv-label",
    "csv-family", "csv-score", "csv-nonfinite", "csv-components", "csv-unavailable-correction", "history-short", "late-tie",
    "membership-hash", "membership-test-id"])
def test_invalid_or_incomplete_evidence_fails_closed(report, completed, mutation):
    fixture = completed
    control_binding, manifest_binding = copy.deepcopy(fixture["control_binding"]), copy.deepcopy(fixture["manifest_binding"])
    root = copy.deepcopy(fixture["completion"])
    if mutation == "root-hash":
        control_binding["sha256"] = "0" * 64
    elif mutation == "manifest-hash":
        manifest_binding["sha256"] = "0" * 64
    elif mutation.startswith("root-"):
        root[{"root-schema": "schema_version", "root-off": "arm", "root-passed": "passed"}[mutation]] = {
            "root-schema": "other", "root-off": "on", "root-passed": False}[mutation]
        control_binding = write(fixture["directory"] / "control.json", root)
    elif mutation.startswith("replay-") or mutation == "missing-replay":
        path = Path(root["fresh_replay"]["path"])
        if mutation == "missing-replay":
            path.unlink()
        else:
            value = json.loads(path.read_text())
            if mutation == "replay-passed": value["passed"] = False
            if mutation == "replay-order": value["image_ids"].reverse()
            if mutation == "replay-identity": value["identities"]["source_sha256"] = "0" * 64
            if mutation == "replay-summary": value["summary"]["accuracy"] = .4
            if mutation == "replay-flag-only": value = {"passed": True}
            root["fresh_replay"] = write(path, value)
            control_binding = write(fixture["directory"] / "control.json", root)
    elif mutation == "membership-hash":
        Path(fixture["manifest"]["authorities"]["records"]["path"]).write_text("modified metadata")
    else:
        train = copy.deepcopy(fixture["train"])
        if mutation == "train-incomplete": train["epochs_completed"] = 99
        if mutation == "train-test": train["role"] = "test"
        if mutation == "train-on": train["arm"] = "on"
        if mutation == "identity": train["identities"]["source_sha256"] = "0" * 64
        if mutation == "checkpoint": train["checkpoint"]["sha256"] = "0" * 64
        if mutation == "late-tie": train["selected_epoch"] = 2
        if mutation.startswith("csv-") or mutation == "membership-test-id":
            rows = copy.deepcopy(fixture["predictions"])
            if mutation == "csv-order": rows.reverse()
            if mutation == "csv-label": rows[0]["label"] = 1
            if mutation == "csv-family": rows[111]["attack_family"] = "copy"
            if mutation == "csv-score": rows[0]["attack_probability"] = .9
            if mutation == "csv-nonfinite": rows[0]["query_logit"] = float("nan")
            if mutation == "csv-components": rows[111]["correction_logit"] = 2.
            if mutation == "csv-unavailable-correction": rows[0].update(query_logit=-3., correction_logit=1.)
            if mutation == "membership-test-id": rows[0]["image_id"] = "test0000"
            train["predictions"] = write_csv(fixture["directory"] / "val_predictions.csv", rows)
        if mutation == "history-short":
            path = fixture["directory"] / "log.csv"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        root["train_result"] = write(fixture["directory"] / "train.json", train)
        control_binding = write(fixture["directory"] / "control.json", root)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.build_report(control_binding, manifest_binding)


def test_explicit_archive_relocation_preserves_authority_bytes(report, completed, tmp_path):
    archived = tmp_path / "copied metadata"
    shutil.copytree(completed["base"], archived)
    original = completed["base"]
    control = dict(completed["control_binding"], path=str(archived / "run/training/off/control.json"))
    manifest = dict(completed["manifest_binding"], path=str(archived / "run/input_manifest.json"))
    shutil.rmtree(original)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.build_report(control, manifest)
    result = report.build_report(control, manifest, relocations=[(str(original), str(archived))])
    assert result["competence"]["passed"] is True
    assert result["identities"] == completed["train"]["identities"]
    assert binding(archived / "run/training/off/control.json")["sha256"] == control["sha256"]


def test_ties_are_genuine_and_query_sigmoid_method_is_labeled(report, completed):
    for row in completed["predictions"]:
        row.update(query_logit=0., correction_logit=row["attack_logit"])
        if not row["context_available"]:
            row.update(attack_logit=0., correction_logit=0., attack_probability=.5, bonafide_score=.5)
    refresh(completed)
    result = report.build_report(completed["control_binding"], completed["manifest_binding"])
    assert result["query_component_only"]["accuracy"] == .5
    assert result["query_component_only"]["auc"] == .5
    assert result["query_component_only"]["genuine_false_positive_rate"] == 0.
    assert result["query_score_method"] == "CPU torch.float32 sigmoid(query_logit); bonafide_score=1-probability; ties predict genuine"


def test_cli_creates_only_new_report_and_preserves_inputs(report, completed, tmp_path):
    before = {path: path.read_bytes() for path in completed["base"].rglob("*") if path.is_file()}
    output = tmp_path / "report.json"
    arguments = ["--control", completed["control_binding"]["path"], "--control-sha256", completed["control_binding"]["sha256"],
                 "--manifest", completed["manifest_binding"]["path"], "--manifest-sha256", completed["manifest_binding"]["sha256"], "--out", str(output)]
    report.main(arguments)
    assert json.loads(output.read_text())["status"] == "complete"
    assert "not an independently trained" in output.with_suffix(".md").read_text()
    with pytest.raises(ValueError, match="new output"):
        report.main(arguments)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_failed_competence_is_reported_not_a_training_gate(report, completed):
    for row in completed["predictions"][111:121]:
        probability = torch.tensor(-2., dtype=torch.float32).sigmoid().item()
        row.update(attack_logit=-2., correction_logit=0., attack_probability=probability, bonafide_score=1. - probability)
    refresh(completed)
    result = report.build_report(completed["control_binding"], completed["manifest_binding"])
    assert result["status"] == "complete"
    assert result["competence"]["passed"] is False
    assert result["combined"]["accuracy"] == 212 / 222
    assert result["combined"]["n_images"] == 222


def test_fresh_replay_accepts_existing_helper_numeric_tolerance(report, completed):
    root = copy.deepcopy(completed["completion"])
    path = Path(root["fresh_replay"]["path"])
    replay = json.loads(path.read_text())
    replay["predictions"][10]["attack_logit"] += 1e-7
    replay["predictions"][10]["query_logit"] += 1e-7
    replay["replay"] = compare_replay(completed["predictions"], replay["predictions"])
    replay["max_abs_difference"] = replay["replay"]["max_abs_difference"]
    root["fresh_replay"] = write(path, replay)
    root_binding = write(completed["directory"] / "control.json", root)
    result = report.build_report(root_binding, completed["manifest_binding"])
    assert result["status"] == "complete"


def test_explicit_metadata_csv_copies_keep_original_hash_authorities(report, completed, tmp_path):
    local = tmp_path / "csv metadata"
    local.mkdir()
    metadata = {}
    for name in ("records", "split"):
        original = Path(completed["manifest"]["authorities"][name]["path"])
        path = local / original.name
        shutil.copyfile(original, path)
        original.unlink()
        metadata[name] = str(path)
    result = report.build_report(completed["control_binding"], completed["manifest_binding"], metadata_paths=metadata)
    assert result["combined"]["n_images"] == 222
    Path(metadata["records"]).write_text("wrong metadata")
    with pytest.raises(ValueError, match="SHA256"):
        report.build_report(completed["control_binding"], completed["manifest_binding"], metadata_paths=metadata)


def test_observed_gpu_sigmoid_rounding_preserves_saved_combined_scores(report, completed):
    logit = 3.2272820472717285
    saved_probability = 0.9618480801582336
    cpu_probability = torch.tensor(logit, dtype=torch.float32).sigmoid().item()
    assert abs(saved_probability - cpu_probability) > 1e-7
    completed["predictions"][111].update(attack_logit=logit, query_logit=logit, correction_logit=0.,
                                         attack_probability=saved_probability, bonafide_score=1. - saved_probability)
    refresh(completed)
    saved_csv = (completed["directory"] / "val_predictions.csv").read_bytes()
    result = report.build_report(completed["control_binding"], completed["manifest_binding"])
    assert result["combined"]["accuracy"] == result["combined"]["auc"] == 1.
    assert result["per_image"][111]["combined_bonafide_score"] == 1. - saved_probability
    assert result["per_image"][111]["query_bonafide_score"] == 1. - cpu_probability
    assert result["per_image"][111]["combined_bonafide_score"] != result["per_image"][111]["query_bonafide_score"]
    assert (completed["directory"] / "val_predictions.csv").read_bytes() == saved_csv


@pytest.mark.parametrize("field", ["attack_probability", "bonafide_score"])
def test_substantive_saved_score_corruption_still_rejected(report, completed, field):
    completed["predictions"][111][field] += 1e-4
    refresh(completed)
    with pytest.raises(ValueError):
        report.build_report(completed["control_binding"], completed["manifest_binding"])
