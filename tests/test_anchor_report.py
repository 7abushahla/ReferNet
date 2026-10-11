import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def report():
    path = ROOT / "scripts/report_anchor_pilots.py"
    assert path.is_file(), "anchor report is not implemented"
    spec = importlib.util.spec_from_file_location("anchor_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.write_text(json.dumps(value))


def csv_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def selected_run(report, tmp_path):
    from refer_or_clear.anchor_contract import contract_for
    from refer_or_clear.pilots import OFFICIAL_WEIGHT_SHA256, json_sha256, summarize_predictions
    run = tmp_path / "pilot"
    (run / "checkpoints").mkdir(parents=True)
    config = {"schema_version": "anchor_pilot_v1", "track": "D", "architecture_id": contract_for("D").architecture_id,
              "data_contract": "legacy_geometry_diagnostic", "seed": 42, "fold": 0, "epochs": 100}
    identities = {key: "a" * 64 for key in ("source_sha256", "records_sha256", "split_sha256")}
    identities.update(config_sha256=json_sha256(config), weights_sha256=OFFICIAL_WEIGHT_SHA256["D"])
    predictions = []
    for index in range(222):
        label = int(index >= 111)
        logit = 2. if label else -2.
        probability = torch.tensor(logit).sigmoid().item()
        predictions.append({"image_id": f"val{index:04d}", "label": label, "design_id": "design",
                            "attack_family": "text_copy" if index >= 206 else ("text_inpaint" if label else ""),
                            "attack_logit": logit, "attack_probability": probability, "bonafide_score": 1. - probability})
    summary = summarize_predictions(predictions)
    summary.update(role="val", selected_epoch=2, image_loss=torch.nn.functional.softplus(torch.tensor(-2.)).item())
    provenance = {"requested_initialization": "levir", "checkpoint_path": "original.pth",
                  "checkpoint_sha256": identities["weights_sha256"], "loaded_keys": ["encoder.weight"],
                  "missing_keys": [], "unexpected_keys": [], "pretrained_tensor_sha256": "b" * 64}
    checkpoint = {"schema_version": "anchor_pilot_v1", "architecture_id": config["architecture_id"], "config": config,
                  "config_sha256": identities["config_sha256"], "identities": identities, "epoch": 2,
                  "selection_key": [1., 1.], "phase": "train", "state_dict": {}, "initialization_provenance": provenance}
    torch.save(checkpoint, run / "checkpoints/best.pt")
    checkpoint_sha = report.file_sha256(run / "checkpoints/best.pt")
    info = {"schema_version": "anchor_pilot_v1", "track": "D", "architecture_id": config["architecture_id"],
            "phase": "train", "config": config, "identities": identities, "paths": {}, "passed": True,
            "initialization": {"passed": True, "provenance": provenance}, "fresh_initialization": {"passed": True},
            "inference_before": {"passed": True}, "inference_after": {"passed": True, "companion_invariance": None},
            "selected_epoch": 2, "selection_key": [1., 1.], "epochs_completed": 100,
            "n_images": {"train": 1778, "val": 222}, "n_active_parameters": 10, "n_total_parameters": 12,
            "no_test_open": True, "opened_roles": ["train", "val"], "selected_checkpoint_sha256": checkpoint_sha,
            "membership": {"val": [{key: row[key] for key in ("image_id", "label", "design_id", "attack_family")}
                                   for row in predictions]}}
    dump(run / "train.json", info)
    dump(run / "val_summary.json", summary)
    csv_rows(run / "val_predictions.csv", predictions)
    csv_rows(run / "log.csv", [{"epoch": epoch, "val_accuracy": .9 if epoch == 1 else 1., "val_auc": 1.,
                                "val_image_loss": summary["image_loss"], "loss": .1} for epoch in range(1, 101)])
    dump(run / "gate.json", {"schema_version": "anchor_pilot_v1", "phase": "selected", "passed": True,
                              "identities": identities, "fresh_initialization": info["fresh_initialization"], "failures": []})
    dump(run / "replay.json", {"passed": True, "fresh_process": True, "identities": identities,
                                "checkpoint_sha256": checkpoint_sha, "inference": {"passed": True}})
    return run


def test_selected_validation_recomputed_and_runtime_not_measured(report, selected_run):
    result = report.inspect_run(selected_run, "D")
    assert result["status"] == "complete", result["reasons"]
    assert result["selected_epoch"] == 2
    assert result["metrics"]["accuracy"] == 1.
    assert result["metrics"]["subgroups"]["copy"]["n_altered"] == 16
    assert result["runtime"]["status"] == "not-measured"
    assert "compound" in result["interpretation"]
    assert "16" in result["copy_uncertainty"]
    assert result["gates"]["companion_invariance"] is None


def test_canonical_prediction_taxonomy_keeps_human_subgroups(report, selected_run):
    rows = report.read_csv(selected_run / "val_predictions.csv")
    membership = json.loads((selected_run / "train.json").read_text())["membership"]["val"]
    assert {row["attack_family"] for row in rows if row["label"] == "1"} == {"text_copy", "text_inpaint"}
    metrics = report.prediction_metrics(rows, membership, "D")
    assert set(metrics["subgroups"]) == {"copy", "inpainting"}
    assert metrics["subgroups"]["copy"] == {"n_genuine": 111, "n_altered": 16, "auc": 1., "accuracy": 1.}
    assert metrics["subgroups"]["inpainting"] == {"n_genuine": 111, "n_altered": 95, "auc": 1., "accuracy": 1.}


def test_report_rejects_unrecognized_canonical_family(report, selected_run):
    rows = report.read_csv(selected_run / "val_predictions.csv")
    membership = json.loads((selected_run / "train.json").read_text())["membership"]["val"]
    rows[111]["attack_family"] = membership[111]["attack_family"] = "face_swap"
    with pytest.raises(ValueError, match="unknown altered family"):
        report.prediction_metrics(rows, membership, "D")


@pytest.mark.parametrize("mutation", ["test", "unknown", "reference", "wrong_epoch", "tie", "metric", "label", "replay", "gate_identity", "gate_failed"])
def test_untrustworthy_evidence_cannot_be_candidate(report, selected_run, mutation):
    info = json.loads((selected_run / "train.json").read_text())
    summary = json.loads((selected_run / "val_summary.json").read_text())
    if mutation == "test":
        summary["role"] = "test"
    elif mutation == "unknown":
        info["schema_version"] = "unknown"
    elif mutation == "reference":
        info["config"]["reference"] = "median"
    elif mutation == "wrong_epoch":
        summary["selected_epoch"] = 100
    elif mutation == "tie":
        history = list(csv.DictReader((selected_run / "log.csv").open()))
        history[0]["val_accuracy"] = "1.0"
        csv_rows(selected_run / "log.csv", history)
    elif mutation == "metric":
        summary["auc"] = .1
    elif mutation == "label":
        rows = list(csv.DictReader((selected_run / "val_predictions.csv").open()))
        rows[0]["label"] = "1"
        csv_rows(selected_run / "val_predictions.csv", rows)
    elif mutation == "replay":
        (selected_run / "replay.json").unlink()
    else:
        gate = json.loads((selected_run / "gate.json").read_text())
        if mutation == "gate_identity":
            gate["identities"]["source_sha256"] = "f" * 64
        else:
            gate["passed"] = False
        dump(selected_run / "gate.json", gate)
    dump(selected_run / "train.json", info)
    dump(selected_run / "val_summary.json", summary)
    result = report.inspect_run(selected_run, "D")
    assert result["status"] != "complete"
    assert result["eligible_candidate"] is False


def test_incomplete_report_preserves_status_and_has_no_winner(report, selected_run, tmp_path):
    result = report.build_report([tmp_path / "missingR", selected_run, tmp_path / "missingN"])
    assert result["status"] == "incomplete"
    assert result["winner"] is None
    assert result["tracks"]["R"]["status"] == "incomplete"
    assert result["tracks"]["D"]["status"] == "complete"
    assert "Protocol B" in result["limitations"]
    assert result["support_readiness"] == "not-assessed"


def test_report_rejects_mixed_run_sources(report, monkeypatch):
    def inspect(path, track):
        return {"status": "complete", "eligible_candidate": True, "reasons": [],
                "identities": {"source_sha256": track, "records_sha256": "a", "split_sha256": "b"},
                "validation_membership_sha256": "same", "runtime": {"status": "not-measured"}}
    monkeypatch.setattr(report, "inspect_run", inspect)
    result = report.build_report(["r", "d", "n"])
    assert result["status"] == "incomplete" and result["winner"] is None
    assert all(value["status"] == "invalid" and not value["eligible_candidate"] for value in result["tracks"].values())


def test_unmeasured_or_foreign_runtime_cannot_be_presented_as_measured(report, selected_run, tmp_path):
    runtime = tmp_path / "runtime.json"
    dump(runtime, {"schema_version": "anchor_cpu_v1", "status": "not-comparable"})
    result = report.build_report([tmp_path / "r", selected_run, tmp_path / "n"], runtime)
    assert result["runtime_status"] == "not-comparable"
    assert result["tracks"]["D"]["runtime"]["status"] == "not-comparable"


def test_missing_runtime_remains_not_measured(report, selected_run, tmp_path):
    result = report.build_report([tmp_path / "r", selected_run, tmp_path / "n"], tmp_path / "absent-runtime.json")
    assert result["runtime_status"] == "not-measured"
    assert result["tracks"]["D"]["runtime"]["status"] == "not-measured"


def test_report_cli_writes_json_and_readable_selected_validation(report, selected_run, tmp_path):
    output = tmp_path / "report.json"
    result = report.main(["--runs", str(tmp_path / "r"), str(selected_run), str(tmp_path / "n"), "--out", str(output)])
    assert json.loads(output.read_text())["winner"] is None
    text = output.with_suffix(".md").read_text()
    assert "selected-validation" in text and "Genuine false positives" in text and "copy" in text
    assert result["status"] == "incomplete"


def test_report_never_overwrites_existing_input(report, selected_run, tmp_path):
    source = selected_run / "train.json"
    original = source.read_bytes()
    with pytest.raises(ValueError, match="new output"):
        report.main(["--runs", str(tmp_path / "r"), str(selected_run), str(tmp_path / "n"), "--out", str(source)])
    assert source.read_bytes() == original


@pytest.mark.parametrize("field,value", [("n_total_parameters", 12.0), ("n_total_parameters", 12.5),
                                          ("n_total_parameters", True), ("n_total_parameters", None),
                                          ("n_active_parameters", True), ("n_active_parameters", 10.0)])
def test_parameter_counts_must_be_integer_not_boolean(report, selected_run, field, value):
    info = json.loads((selected_run / "train.json").read_text())
    info[field] = value
    dump(selected_run / "train.json", info)
    result = report.inspect_run(selected_run, "D")
    assert result["status"] == "invalid" and not result["eligible_candidate"]
    assert any("invalid parameter counts" in reason for reason in result["reasons"])
