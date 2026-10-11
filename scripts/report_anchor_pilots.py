"""Report selected source-validation evidence only, without selecting a family."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import pickle
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.anchor_contract import SCHEMA_VERSION, contract_for
from refer_or_clear.pilots import (
    _validate_provenance, file_sha256, json_sha256, read_pilot_checkpoint,
    summarize_predictions, validate_weight_request,
)
from refer_or_clear.train import selection_key

LIMITATIONS = (
    "R/D/N are compound recipe contrasts, not backbone-only or causal comparisons. "
    "D/N are legacy_geometry_diagnostic and D is not historical-stream-matched. "
    "The fold-0 copy subset has only 16 altered images and cannot establish superiority. "
    "A complete passing query candidate remains provisional pending a matched reference comparison. "
    "Protocol B and test evidence, reference-arm results, and FantasyID audits are not selection inputs. "
    "No sweep or later-protocol evaluation is authorized by this report. "
    "Preliminary cluster CPU timing is not end-to-end officer-device latency."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_metrics(expected, actual, name):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{name} missing metrics")
        for key, value in expected.items():
            require(key in actual, f"{name}.{key} missing")
            same_metrics(value, actual[key], f"{name}.{key}")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6), f"{name} metric mismatch")
    else:
        require(expected == actual, f"{name} metric mismatch")


def selected_history(info, history):
    require(len(history) == 100, "incomplete 100-epoch curves")
    require([int(row["epoch"]) for row in history] == list(range(1, 101)), "invalid epoch sequence")
    best_key, best_row = None, None
    for row in history:
        accuracy, auc = float(row["val_accuracy"]), float(row["val_auc"])
        loss = float(row["val_image_loss"])
        require(all(math.isfinite(value) for value in (accuracy, auc, loss))
                and 0 <= accuracy <= 1 and 0 <= auc <= 1 and loss >= 0, "invalid validation curves")
        key = selection_key("accuracy", accuracy, auc, loss, float(row.get("loss", 0)))
        if best_key is None or key > best_key:
            best_key, best_row = key, row
    require(int(best_row["epoch"]) == info["selected_epoch"], "accuracy-first/earlier-tie selected epoch mismatch")
    same_metrics(list(best_key), info["selection_key"], "selected key")
    return best_row


def prediction_metrics(rows, membership, track):
    require(len(rows) == 222 and len(membership) == 222, "incomplete official validation predictions")
    ids = [row["image_id"] for row in rows]
    require(len(set(ids)) == 222 and ids == [row["image_id"] for row in membership], "prediction membership mismatch")
    predictions = []
    for row, member in zip(rows, membership):
        require(row.get("role", "val") == "val", "mixed test/validation predictions")
        require(row["label"] in ("0", "1"), "prediction label must be binary")
        parsed = dict(row, label=int(row["label"]))
        for key in ("label", "image_id", "design_id", "attack_family"):
            require(parsed[key] == member[key], f"prediction {key} differs from validation membership")
        require(parsed["label"] == 0 or parsed["attack_family"] in ("text_copy", "text_inpaint"), "unknown altered family")
        for key in ("attack_logit", "attack_probability", "bonafide_score"):
            parsed[key] = float(row[key])
            require(math.isfinite(parsed[key]), "non-finite prediction")
        probability = torch.tensor(parsed["attack_logit"], dtype=torch.float64).sigmoid().item()
        require(math.isclose(probability, parsed["attack_probability"], abs_tol=1e-6, rel_tol=1e-6)
                and math.isclose(1. - probability, parsed["bonafide_score"], abs_tol=1e-6, rel_tol=1e-6),
                "inconsistent logit/probability/bonafide score")
        if track == "R":
            difference = float(row["class_logit_1"]) - float(row["class_logit_0"])
            require(math.isclose(difference, parsed["attack_logit"], rel_tol=1e-5, abs_tol=1e-5), "R class logits disagree")
        predictions.append(parsed)
    result = summarize_predictions(predictions)
    logits = torch.tensor([row["attack_logit"] for row in predictions], dtype=torch.float64)
    labels = torch.tensor([row["label"] for row in predictions], dtype=torch.float64)
    result["image_loss"] = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    return result


def inspect_run(run_dir, track):
    run_dir = Path(run_dir)
    result = {"track": track, "run_dir": str(run_dir), "status": "incomplete", "eligible_candidate": False,
              "reasons": [], "runtime": {"status": "not-measured"},
              "interpretation": "compound recipe contrast; no causal or optimality conclusion",
              "copy_uncertainty": "Only 16 fold-0 copy forgeries; subgroup evidence is limited."}
    try:
        info, summary = read_json(run_dir / "train.json"), read_json(run_dir / "val_summary.json")
        contract, config = contract_for(track), info["config"]
        require(info["schema_version"] == SCHEMA_VERSION and config["schema_version"] == SCHEMA_VERSION, "unknown schema")
        require(info["phase"] == "train" and summary["role"] == "val", "only selected long-run validation is permitted")
        require(info["track"] == config["track"] == track and info["architecture_id"] == config["architecture_id"]
                == contract.architecture_id and config["data_contract"] == contract.data_contract, "unknown/mixed/reference architecture")
        require(config.get("reference") in (None, "none") and config.get("protocol", "A") == "A", "reference/Protocol B inputs forbidden")
        require(config["fold"] == 0 and config["seed"] == 42 and config["epochs"] == 100
                and info["epochs_completed"] == 100, "incomplete/unapproved pilot recipe")
        require(info["no_test_open"] is True and info["opened_roles"] == ["train", "val"], "forbidden test role access")
        require(info["n_images"] == {"train": 1778, "val": 222}, "incomplete official membership")
        identities = info["identities"]
        for key in ("source_sha256", "config_sha256", "records_sha256", "split_sha256", "weights_sha256"):
            value = identities[key]
            require(len(value) == 64 and all(character in "0123456789abcdef" for character in value), "invalid identity hash")
        require(identities["config_sha256"] == json_sha256(config), "effective config identity mismatch")
        validate_weight_request(track, identities["weights_sha256"])
        _validate_provenance(info["initialization"]["provenance"], track, identities["weights_sha256"])
        history = selected_history(info, read_csv(run_dir / "log.csv"))
        require(summary["selected_epoch"] == info["selected_epoch"], "summary selected epoch mismatch")
        metrics = prediction_metrics(read_csv(run_dir / "val_predictions.csv"), info["membership"]["val"], track)
        same_metrics(metrics, summary, "selected validation")
        same_metrics({"accuracy": float(history["val_accuracy"]), "auc": float(history["val_auc"]),
                      "image_loss": float(history["val_image_loss"])}, metrics, "selected curve")
        checkpoint_path = run_dir / "checkpoints/best.pt"
        checkpoint_sha = file_sha256(checkpoint_path)
        require(checkpoint_sha == info["selected_checkpoint_sha256"], "checkpoint SHA256 mismatch")
        checkpoint = read_pilot_checkpoint(checkpoint_path, config, identities)
        require(checkpoint["phase"] == "train" and checkpoint["epoch"] == info["selected_epoch"]
                and checkpoint["selection_key"] == info["selection_key"], "selected checkpoint epoch/key mismatch")
        require(checkpoint["initialization_provenance"] == info["initialization"]["provenance"], "checkpoint initialization mismatch")
        gate, replay = read_json(run_dir / "gate.json"), read_json(run_dir / "replay.json")
        require(gate.get("schema_version") == SCHEMA_VERSION and gate.get("phase") == "selected", "missing selected gate")
        require(gate.get("identities") == replay.get("identities") == identities, "gate/replay identity mismatch")
        require(gate.get("fresh_initialization") == info["fresh_initialization"], "gate pristine initialization mismatch")
        require(replay.get("checkpoint_sha256") == checkpoint_sha, "replay checkpoint mismatch")
        passed = (info.get("passed") is True and gate.get("passed") is True and not gate.get("failures")
                  and replay.get("passed") is True and replay.get("fresh_process") is True
                  and replay.get("inference", {}).get("passed") is True
                  and all(info.get(key, {}).get("passed") is True for key in
                          ("initialization", "fresh_initialization", "inference_before", "inference_after")))
        require(type(info["n_active_parameters"]) is int and type(info["n_total_parameters"]) is int
                and 0 < info["n_active_parameters"] <= info["n_total_parameters"],
                "invalid parameter counts")
        result.update(status="complete" if passed else "failed", eligible_candidate=passed,
                      selected_epoch=info["selected_epoch"], metrics=metrics, identities=identities,
                      data_contract=contract.data_contract, validation_membership_sha256=json_sha256(info["membership"]["val"]),
                      n_active_parameters=info["n_active_parameters"], n_total_parameters=info["n_total_parameters"],
                      checkpoint_bytes=checkpoint_path.stat().st_size,
                      gates={"initialization": info["initialization"]["passed"], "fresh_process_replay": replay.get("passed") is True
                             and replay.get("fresh_process") is True, "invariance": info["inference_after"].get("passed") is True,
                             "companion_invariance": info["inference_after"].get("companion_invariance")},
                      acceptance_targets_met=metrics["auc"] >= .99 and metrics["accuracy"] >= .97,
                      artifact_sha256={name: file_sha256(run_dir / name) for name in
                                       ("train.json", "val_summary.json", "val_predictions.csv", "log.csv", "gate.json", "replay.json")})
        if not passed:
            result["reasons"].append("required initialization/replay/invariance gate failed")
    except FileNotFoundError as error:
        result["reasons"].append(f"missing evidence ({error.filename})")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        result.update(status="invalid", eligible_candidate=False)
        result["reasons"].append(str(error))
    return result


def build_report(runs, runtime=None):
    require(len(runs) == 3, "provide R D N runs in fixed order")
    tracks = {track: inspect_run(path, track) for track, path in zip("RDN", runs)}
    complete = [value for value in tracks.values() if value["status"] == "complete"]
    if complete:
        first = complete[0]
        for value in complete[1:]:
            if (any(value["identities"][key] != first["identities"][key]
                    for key in ("source_sha256", "records_sha256", "split_sha256"))
                    or value["validation_membership_sha256"] != first["validation_membership_sha256"]):
                for candidate in complete:
                    candidate.update(status="invalid", eligible_candidate=False)
                    candidate["reasons"].append("mixed source/validation identities across tracks")
                break
    runtime_status = "not-measured"
    if runtime is not None:
        try:
            measurement = read_json(runtime)
            require(measurement.get("schema_version") == "anchor_cpu_v1" and measurement.get("status") == "measured",
                    "runtime not comparable")
            from bench_anchor_pilots import validate_measurement
            validate_measurement(measurement, tracks)
            for track, candidate in tracks.items():
                candidate["runtime"] = dict(status="measured", **measurement["tracks"][track])
            runtime_status = "measured"
        except FileNotFoundError:
            runtime_status = "not-measured"
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            runtime_status = "not-comparable"
            for candidate in tracks.values():
                candidate["runtime"] = {"status": "not-comparable"}
    return {"schema_version": "anchor_report_v1", "role": "selected_validation", "tracks": tracks,
            "status": "complete" if all(value["status"] == "complete" for value in tracks.values()) else "incomplete",
            "runtime_status": runtime_status, "winner": None, "family_selection": "not-authorized",
            "support_readiness": "not-assessed", "limitations": LIMITATIONS,
            "curves": {track: {"path": str(Path(path) / "log.csv"), "use": "selection verification only"}
                       for track, path in zip("RDN", runs)},
            "historical_test_evidence": {"status": "not-consumed", "use": "separate prior exploratory evidence"}}


def render_markdown(report):
    lines = ["# Anchor pilot selected-validation report", "", report["limitations"], "",
             "| Track | Status | Epoch | AUC | Accuracy | Image loss | Active/total parameters | Runtime |",
             "|---|---|---|---|---|---|---|---|"]
    for track, value in report["tracks"].items():
        metrics = value.get("metrics", {})
        lines.append(f"| {track} | {value['status']} | {value.get('selected_epoch', 'unavailable')} | "
                     f"{metrics.get('auc', 'unavailable')} | {metrics.get('accuracy', 'unavailable')} | "
                     f"{metrics.get('image_loss', 'unavailable')} | {value.get('n_active_parameters', 'unavailable')}/"
                     f"{value.get('n_total_parameters', 'unavailable')} | {value['runtime']['status']} |")
    for track, value in report["tracks"].items():
        lines.extend(["", f"## {track}", "", value["copy_uncertainty"]])
        for reason in value["reasons"]:
            lines.append(f"- {reason}")
        if "metrics" in value:
            metrics = value["metrics"]
            lines.append(f"- Gates {json.dumps(value['gates'], sort_keys=True)}")
            for family, subgroup in metrics["subgroups"].items():
                lines.append(f"- {family} {json.dumps(subgroup, sort_keys=True)}")
            lines.append(f"- Genuine false positives {json.dumps(metrics['genuine_false_positives'])}")
            lines.append(f"- Altered misses {json.dumps(metrics['altered_misses'])}")
    lines.extend(["", "## Decision", "", "No winner is claimed. Support readiness is not assessed by these inputs.",
                  "Matched-reference confirmation and any continuation require a separate decision.", "",
                  "## Curves and historical test evidence", "",
                  "Curves are used only to verify selected epochs. Historical test evidence is not consumed."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True, metavar=("R", "D", "N"))
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    markdown_path = args.out.with_suffix(".md")
    require(args.out != markdown_path and not args.out.exists() and not markdown_path.exists(),
            "new output paths required; existing source/inputs/reports cannot be overwritten")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    result = build_report(args.runs, args.runtime)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with markdown_path.open("x") as stream:
        stream.write(render_markdown(result))
    return result


if __name__ == "__main__":
    main()
