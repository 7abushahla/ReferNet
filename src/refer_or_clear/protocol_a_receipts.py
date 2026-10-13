"""Authenticate Protocol A receipt chains without images, models, or GPU work.

Checkpoint headers are read with the existing CPU, weights-only core reader
after streaming SHA256 authentication. This checks evidence consistency, not
independent proof of GPU execution. The coordinator supplies the external gate
hash. Numeric loss/score auditing uses the existing FP32 replay tolerances;
performance thresholds and selection rules are unchanged.
"""
from __future__ import annotations

import copy
import csv
from io import StringIO
import json
import math
from pathlib import Path

import yaml

from .data import protocol_a_reference as bridge
from .data.direct_reference import _bytes, _json, _path, _require, _verifier
from .direct_reference_pilot import (
    _validate_contract, _validate_initialization, read_direct_reference_checkpoint,
)
from .pilots import ATOL, RTOL, OFFICIAL_WEIGHT_SHA256, _selection_record, json_sha256, summarize_predictions
from .protocol_a_contract import training_accounting
from .train import selection_key


def _service():
    from . import protocol_a_train
    return protocol_a_train


def _sha(value):
    _require(_verifier._sha256_value(value), "invalid receipt SHA256")
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(first, second):
    return _number(first) and _number(second) and abs(first - second) <= ATOL + RTOL * abs(second)


def _count(document, name, expected):
    _require(type(document.get(name)) is int and document[name] == expected, f"receipt {name} mismatch")


def _canonical(binding, path):
    _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "canonical path/SHA256 binding required")
    _sha(binding["sha256"])
    _require(binding["path"] == str(_path(str(path))) and _path(binding["path"]) == _path(str(path)),
             "noncanonical receipt/checkpoint path")
    return binding


def expected_identities(context, config, weights_binding, registration_binding, source_sha256,
                        phase_binding, initialization):
    """Derive core identities from current authorities, not receipt claims."""
    metadata = json.loads(context.identities)
    result = dict(source_sha256=_sha(source_sha256), config_sha256=json_sha256(config),
        weights_sha256=weights_binding["sha256"], records_sha256=metadata["records_sha256"],
        split_sha256=metadata["split_sha256"], enrollment_sha256=metadata["enrollment_manifest_sha256"],
        registration_sha256=registration_binding["sha256"], phase_authority_sha256=phase_binding["sha256"],
        preprocess_sha256=json_sha256(dict(identities=metadata, policy=bridge._policy())),
        initialization_sha256=json_sha256(initialization))
    _validate_contract(config, result)
    _validate_initialization(initialization, config, result)
    _count(initialization, "adapter_seed", config["adapter_seed"])
    _sha(initialization.get("adapter_tensor_sha256"))
    _sha(initialization["base"].get("pretrained_tensor_sha256"))
    _require(initialization["base"].get("checkpoint_path") == str(_path(weights_binding["path"])),
             "initialization weights path mismatch")
    return result


def read_predictions(binding, rows, fused_query_weight=1.):
    """Authenticate scores and join exact ordered source metadata, without pixels.

    ``fused_query_weight`` is the arm recipe's scale on the query logit for rows with
    reference context; rows without context always carry the query logit alone.
    """
    _require(_number(fused_query_weight) and 0. <= fused_query_weight <= 1., "fused_query_weight must be a number in [0, 1]")
    try:
        predictions = list(csv.DictReader(StringIO(_bytes(binding).decode("utf-8"))))
        _require(len(predictions) == len(rows), "prediction population mismatch")
        _require([row["image_id"] for row in predictions] == rows.image_id.tolist(), "prediction ID/order mismatch")
        for prediction, expected in zip(predictions, rows.to_dict("records")):
            prediction["label"] = int(prediction["label"])
            _require(all(prediction[key] == expected[key] for key in ("design_id", "label", "attack_family")),
                     "prediction source metadata mismatch")
            for name in ("attack_logit", "query_logit", "correction_logit", "attack_probability", "bonafide_score"):
                prediction[name] = float(prediction[name])
                _require(_number(prediction[name]), "nonfinite prediction")
            for name in ("context_available", "referral_required"):
                _require(prediction[name] in ("True", "False"), "prediction boolean mismatch")
                prediction[name] = prediction[name] == "True"
            logit = prediction["attack_logit"]
            probability = 1. / (1. + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1. + math.exp(logit))
            _require(0 <= prediction["attack_probability"] <= 1 and 0 <= prediction["bonafide_score"] <= 1
                     and _close(prediction["attack_probability"], probability)
                     and _close(prediction["bonafide_score"], 1. - prediction["attack_probability"])
                     and _close(logit, (fused_query_weight if prediction["context_available"] else 1.)
                                * prediction["query_logit"] + prediction["correction_logit"])
                     and prediction["context_available"] is not prediction["referral_required"],
                     "prediction score/logit/referral mismatch")
        return predictions
    except (KeyError, TypeError, UnicodeDecodeError) as error:
        raise ValueError(f"malformed prediction receipt: {error}") from error


def _verify_summary(summary, predictions):
    _require(isinstance(summary, dict), "summary receipt required")
    expected = summarize_predictions(predictions)
    _require(all(type(summary.get(key)) is type(value) and summary[key] == value
                 for key, value in expected.items()), "summary metrics differ from authenticated predictions")
    loss = sum(max(row["attack_logit"], 0.) - row["attack_logit"] * row["label"]
               + math.log1p(math.exp(-abs(row["attack_logit"]))) for row in predictions) / len(predictions)
    _require(_close(summary.get("image_loss"), loss), "summary classification loss mismatch")
    _require(summary.get("buffers_unchanged") is True, "evaluation buffer evidence required")
    available = sum(row["context_available"] for row in predictions)
    _count(summary, "n_context_available", available)
    _count(summary, "n_context_unavailable", len(predictions) - available)
    _require(summary.get("referral_ids") == [row["image_id"] for row in predictions if row["referral_required"]],
             "summary referral population mismatch")


def _request(binding, context, config, weights, source_sha256):
    _canonical(binding, _path(binding["path"]))
    request = _json(binding)
    _require(request.get("schema_version") == "protocol_a_request_v1" and request.get("protocol") == "A"
             and type(request.get("fold")) is int and request["fold"] == context.contract.fold
             and request.get("source_sha256") == source_sha256, "parent request identity mismatch")
    for name, expected in (("records", context.bindings[0]), ("split", context.bindings[1]),
                           ("enrollment", context.bindings[2]),
                           ("weights", weights)):
        _require(request.get(name) == expected, f"parent request {name} binding mismatch")
    relocation_bindings = {}
    for authority in context.bindings[3:]:
        schema = _json(authority).get("schema_version")
        name = {"protocol_a_product_root_v1": "root",
                "protocol_a_metadata_relocation_v1": "metadata_relocation"}.get(schema)
        _require(name is not None and name not in relocation_bindings,
                 "source-only unique A relocation authorities required")
        relocation_bindings[name] = authority
    for name in ("root", "metadata_relocation"):
        _require(request.get(name) == relocation_bindings.get(name), f"parent request {name} authority mismatch")
    _require(request.get("root") is not None or context.product_root == context.manifest["output_root"],
             "relocated products require the parent root authority")
    _require(_path(request["raw_root"]) == context.raw_root, "parent request raw root mismatch")
    supplied = yaml.safe_load(_bytes(request["config"]))
    _require(_service().effective_config(supplied, context, False) == _service().effective_config(config, context, False),
             "parent request config mismatch")
    root = _path(request["output_root"])
    for protected in (context.raw_root, _path(context.manifest["output_root"]), _path(context.product_root)):
        _require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                 "receipt output overlaps protected inputs")
    return request, root


def _checkpoint(binding, path, config, identities, phase, epoch, key, initialization):
    _canonical(binding, path)
    observed = _verifier._read_file(_path(binding["path"]))[0]
    _require(observed == binding["sha256"], "selected checkpoint SHA256 mismatch")
    checkpoint = read_direct_reference_checkpoint(path, config, identities)
    _require(checkpoint.get("schema_version") == "direct_reference_pilot_v1"
             and checkpoint.get("architecture_id") == config["architecture_id"]
             and checkpoint.get("arm") == config["arm"] and json_sha256(checkpoint.get("config")) == json_sha256(config)
             and checkpoint.get("config_sha256") == json_sha256(config)
             and checkpoint.get("identities") == identities and checkpoint.get("phase") == phase
             and checkpoint.get("selected") is True and type(checkpoint.get("epoch")) is int
             and checkpoint["epoch"] == epoch and checkpoint.get("selection_key") == _selection_record(key)
             and checkpoint.get("initialization_provenance") == initialization,
             "selected checkpoint header/initialization mismatch")
    return checkpoint


def _fresh(binding, path, parent_binding, parent, request_binding, identities, predictions):
    _canonical(binding, path)
    report = _json(binding)
    _require(report.get("passed") is True and report.get("fresh_process") is True
             and report.get("no_test_open") is True and type(report.get("pid")) is int and report["pid"] > 0
             and report["pid"] != parent["pid"] and report.get("parent_result") == parent_binding
             and report.get("request") == request_binding and report.get("checkpoint") == parent["checkpoint"]
             and report.get("identities") == identities, "fresh replay parent/process/identity mismatch")
    _count(report, "n_images", len(predictions))
    comparison = report.get("comparison", {})
    _require(comparison.get("passed") is True and _number(comparison.get("atol")) and comparison["atol"] == ATOL
             and _number(comparison.get("rtol")) and comparison["rtol"] == RTOL
             and _number(comparison.get("max_abs_difference")) and comparison["max_abs_difference"] >= 0,
             "replay logit comparison evidence mismatch")
    _count(comparison, "n_images", len(predictions))
    _require(comparison["max_abs_difference"] <= ATOL + RTOL * max(abs(row["attack_logit"]) for row in predictions),
             "replay maximum logit difference exceeds declared tolerance")
    _verify_summary(report.get("summary"), predictions)
    return report


def validate_training_gate(gate_binding, context, config, weights_binding, registration_binding,
                           source_sha256, *, request_binding=None):
    """Authenticate both actual smoke/fresh-replay chains before full training."""
    try:
        shared = _service().effective_config(config, context, False)
        _service()._rows(context)
        _require(weights_binding.get("sha256") == OFFICIAL_WEIGHT_SHA256["D_running_pilot"], "official weights required")
        metadata = json.loads(context.identities)
        expected = dict(protocol="A", fold=context.contract.fold, source_sha256=_sha(source_sha256),
            records_sha256=metadata["records_sha256"], split_sha256=metadata["split_sha256"],
            enrollment_sha256=metadata["enrollment_manifest_sha256"], weights_sha256=weights_binding["sha256"],
            registration_sha256=_sha(registration_binding["sha256"]), recipe_sha256=json_sha256(shared))
        gate = _json(gate_binding)
        _require(gate.get("schema_version") == "protocol_a_training_gate_v1"
                 and all(type(gate.get(key)) is type(value) and gate[key] == value for key, value in expected.items()),
                 "same-source gate identity mismatch")
        _require(all(gate.get(key) is True for key in ("passed", "no_test_open", "both_arm_smoke", "replay", "batch_invariance")),
                 "successful both-arm gate required")
        if request_binding is not None:
            _require(gate.get("request") == request_binding, "gate parent request differs from external authority")
        request_binding = copy.deepcopy(gate["request"])
        request, root = _request(request_binding, context, config, weights_binding, source_sha256)
        _canonical(gate_binding, root / "gate/gate.json")
        _require(gate.get("registration") == registration_binding, "gate registration authority mismatch")
        _canonical(registration_binding, root / "registration.json")
        _bytes(registration_binding)
        for binding in context.bindings:
            _bytes(binding)
        arms = gate.get("arm_reports")
        _require(isinstance(arms, dict) and set(arms) == {"off", "on"}, "both authenticated arm reports required")
        rows = {role: context.rows(role).head(32) for role in ("train", "val")}
        _require(all(len(frame) == 32 for frame in rows.values()), "32 source training/validation rows required")
        initial_states = []
        initial_provenance = []
        for arm, enabled in (("off", False), ("on", True)):
            directory = root / "gate" / arm
            smoke_binding = arms[arm]["smoke"]
            _canonical(smoke_binding, directory / "smoke.json")
            smoke = _json(smoke_binding)
            effective = _service().effective_config(config, context, enabled)
            ids = expected_identities(context, effective, weights_binding, registration_binding,
                                      source_sha256, request_binding, smoke["initialization"])
            _require(smoke.get("passed") is True and smoke.get("no_test_open") is True
                     and type(smoke.get("pid")) is int and smoke["pid"] > 0
                     and smoke.get("request") == request_binding and json_sha256(smoke.get("config")) == json_sha256(effective)
                     and smoke.get("identities") == ids, "smoke parent/config/current identities mismatch")
            _sha(smoke.get("fresh_initial_state_sha256"))
            initial_states.append(smoke["fresh_initial_state_sha256"])
            initial_provenance.append({name: value for name, value in smoke["initialization"].items()
                                       if name not in ("reference_enabled", "local_enabled", "reference_bn", "fused_query_weight")})
            for role in ("train", "val"):
                _require(smoke.get(f"source_{role}_ids") == rows[role].image_id.tolist(), "smoke exact source membership mismatch")
            stats = smoke["stats"]
            _count(stats, "n_images", 32)
            _count(stats, "optimizer_steps", 4)
            _require(_number(stats.get("loss")) and stats["loss"] >= 0
                     and _loss_consistent(stats, effective["auxiliary_query_weight"]), "classification-only smoke loss evidence mismatch")
            invariant = smoke["batch_invariance"]
            _require(invariant.get("passed") is True and invariant.get("buffers_unchanged") is True
                     and _number(invariant.get("max_abs_difference")) and invariant["max_abs_difference"] >= 0,
                     "stored-buffer companion invariance evidence required")
            _count(invariant, "n_images", 4)
            _canonical(smoke["predictions"], directory / "val_predictions.csv")
            predictions = read_predictions(smoke["predictions"], rows["val"], effective["fused_query_weight"])
            _require(invariant["max_abs_difference"] <= ATOL + RTOL * max(abs(row["attack_logit"]) for row in predictions[:4]),
                     "invariance maximum logit difference exceeds authenticated singleton tolerance")
            _verify_summary(smoke["summary"], predictions)
            key = selection_key("accuracy", smoke["summary"]["accuracy"], smoke["summary"]["auc"],
                                smoke["summary"]["image_loss"], stats["loss"])
            _checkpoint(smoke["checkpoint"], directory / "selected.pt", effective, ids, "smoke", 1, key, smoke["initialization"])
            _fresh(arms[arm]["replay"], directory / "fresh-replay.json", smoke_binding, smoke, request_binding, ids, predictions)
        _require(initial_states[0] == initial_states[1] and initial_provenance[0] == initial_provenance[1],
                 "matched arms must share fresh initialization")
        return gate
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"malformed Protocol A gate receipt: {error}") from error


def _binding(path):
    return dict(path=str(_path(str(path))), sha256=_verifier._read_file(_path(str(path)))[0])


def _loss_consistent(row, auxiliary_query_weight):
    """Fused BCE is ``image_loss``; ``loss`` adds the weighted auxiliary query BCE when configured."""
    if float(auxiliary_query_weight) == 0.:
        return _close(row["loss"], row["image_loss"])
    return _number(row["loss"]) and _number(row["image_loss"]) and row["loss"] >= row["image_loss"] - 1e-6


def _history(binding, path, accounting, auxiliary_query_weight=0.):
    _canonical(binding, path)
    history = list(csv.DictReader(StringIO(_bytes(binding).decode("utf-8"))))
    _require(len(history) == 100, "complete 100-epoch history required")
    best_key, best_epoch = None, None
    for epoch, row in enumerate(history, 1):
        for name, expected in (("epoch", epoch), ("n_images", accounting.images_per_epoch),
                               ("optimizer_steps", accounting.batches_per_epoch)):
            _require(row[name] == str(expected), f"history {name} mismatch")
        for name in ("loss", "image_loss", "val_image_loss", "lr_pretrained", "lr_new"):
            row[name] = float(row[name])
            _require(_number(row[name]) and row[name] >= 0, f"invalid history {name}")
        _require(_loss_consistent(row, auxiliary_query_weight) and row["bce_map"] == row["dice_map"] == "",
                 "classification-only history required")
        for name in ("val_accuracy", "val_auc"):
            row[name] = float(row[name]) if row[name] else None
            _require(row[name] is None or (_number(row[name]) and 0 <= row[name] <= 1), "invalid history metric")
        _require(row["val_accuracy"] is not None, "validation accuracy selection evidence required")
        key = selection_key("accuracy", row["val_accuracy"], row["val_auc"], row["val_image_loss"], row["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
    return history, best_key, best_epoch


def validate_completed_training(context, request, base, arm, output, gate_binding, request_binding, *, require_fresh=True):
    """Authenticate completed source training and its fresh selected replay.

    Returns the authenticated training receipt and current arm-specific config.
    require_fresh=False validates the parent before a fresh child is generated.
    Evaluation must use the default True. Neither mode authorizes allocation,
    training, or target access.
    """
    try:
        _require(arm in ("off", "on") and type(require_fresh) is bool, "OFF/ON arm and boolean replay requirement required")
        config = _service().effective_config(base, context, arm == "on")
        supplied, root = _request(request_binding, context, base, request["weights"], request["source_sha256"])
        _require(request == supplied and _path(str(output)) == root, "current request/output authority mismatch")
        gate = _json(gate_binding)
        registration = gate["registration"]
        gate = validate_training_gate(gate_binding, context, base, request["weights"], registration,
                                      request["source_sha256"], request_binding=request_binding)
        directory = root / "training" / arm
        parent_binding = _binding(directory / "train.json")
        result = _json(parent_binding)
        accounting = training_accounting(context.contract, config)
        expected = dict(schema_version="protocol_a_train_v1", protocol="A", fold=context.contract.fold,
            passed=True, no_test_open=True, phase="train", diagnostic_only=False,
            arm=arm, reference_enabled=arm == "on", role="val", epochs_completed=100,
            optimizer_steps=accounting.total_optimizer_steps, n_train=accounting.n_train, n_val=accounting.n_val,
            batches_per_epoch=accounting.batches_per_epoch, images_per_epoch=accounting.images_per_epoch)
        _require(all(type(result.get(name)) is type(value) and result[name] == value for name, value in expected.items()),
                 "completed training contract/accounting mismatch")
        _require(type(result.get("pid")) is int and result["pid"] > 0 and result.get("request") == request_binding
                 and result.get("registration") == registration and result.get("diagnostic_gate") == gate_binding,
                 "completed parent/process/input binding mismatch")
        scheduler = dict(total_updates=accounting.total_optimizer_steps, warmup_updates=accounting.effective_warmup_iters)
        _require(result.get("scheduler") == scheduler
                 and all(type(result["scheduler"].get(name)) is int for name in scheduler), "scheduler accounting mismatch")
        smoke = _json(gate["arm_reports"][arm]["smoke"])
        _sha(result.get("fresh_initial_state_sha256"))
        _require(result["fresh_initial_state_sha256"] == smoke["fresh_initial_state_sha256"]
                 and result.get("initialization") == smoke["initialization"], "completed fresh initialization differs from smoke")
        ids = expected_identities(context, config, request["weights"], registration, request["source_sha256"],
                                  gate_binding, result["initialization"])
        _require(result.get("identities") == ids, "completed identities differ from current authorities")
        history, key, epoch = _history(result["history"], directory / "log.csv", accounting,
                                       config["auxiliary_query_weight"])
        _count(result, "selected_epoch", epoch)
        _require(result.get("selection_key") == _selection_record(key), "earliest accuracy/AUC selection mismatch")
        omissions = result["omitted_train_ids"]
        _require(isinstance(omissions, list) and len(omissions) == 100, "complete drop-last omission evidence required")
        train_ids = set(context.rows("train").image_id)
        for number, omission in enumerate(omissions, 1):
            _count(omission, "epoch", number)
            values = omission.get("image_ids")
            _require(isinstance(values, list) and all(isinstance(value, str) for value in values)
                     and len(values) == accounting.n_train - accounting.images_per_epoch
                     and len(set(values)) == len(values) and set(values) <= train_ids, "invalid training omissions")
        _canonical(result["predictions"], directory / "val_predictions.csv")
        predictions = read_predictions(result["predictions"], context.rows("val"), config["fused_query_weight"])
        _canonical(result["summary"], directory / "val_summary.json")
        summary = _json(result["summary"])
        _verify_summary(summary, predictions)
        _require(summary.get("role") == "val" and summary.get("protocol") == "A"
                 and type(summary.get("fold")) is int and summary["fold"] == context.contract.fold,
                 "selected summary source role/fold mismatch")
        _count(summary, "selected_epoch", epoch)
        selected = history[epoch - 1]
        _require(summary["accuracy"] == selected["val_accuracy"] and summary["auc"] == selected["val_auc"]
                 and _close(summary["image_loss"], selected["val_image_loss"]), "selected history/prediction metrics mismatch")
        _checkpoint(result["checkpoint"], directory / "best.pt", config, ids, "train", epoch, key, result["initialization"])
        _canonical(result["in_process_replay"], directory / "replay.json")
        replay = _json(result["in_process_replay"])
        _require(result.get("replay") == replay and replay.get("passed") is True
                 and replay.get("fresh_process") is False and replay.get("phase") == "train", "in-process replay evidence mismatch")
        _count(replay, "n_images", accounting.n_val)
        if require_fresh:
            fresh_binding = _binding(directory / "fresh-replay.json")
            _fresh(fresh_binding, directory / "fresh-replay.json", parent_binding, result, request_binding, ids, predictions)
        return result, config
    except (KeyError, TypeError, AttributeError, IndexError, UnicodeDecodeError) as error:
        raise ValueError(f"malformed completed Protocol A training receipt: {error}") from error

