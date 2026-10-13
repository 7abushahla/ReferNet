"""Allocated, source-bound matched Protocol B smoke, training, and evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from anchor_job_manifest import source_entries, object_sha256
from refer_or_clear.data.direct_reference import _bytes, _json, _path, _require
from refer_or_clear.data import protocol_b_reference as bridge
from refer_or_clear import protocol_b_train as service
from refer_or_clear import protocol_b_receipts as receipts
from refer_or_clear import direct_reference_pilot as core
from refer_or_clear.models.direct_reference import build_direct_reference_model
from refer_or_clear.pilots import configure_precision, json_sha256, tensor_sha256, _write_csv, compare_replay, ATOL, RTOL
from refer_or_clear.protocol_b_contract import training_accounting, make_protocol_b_scheduler
from refer_or_clear.train import selection_key


def require_gpu_allocation():
    node = os.environ.get("SLURMD_NODENAME", "").lower()
    _require(os.environ.get("SLURM_JOB_ID", "").isdigit() and node
             and not any(token in node for token in ("login", "head", "frontend", "lgn"))
             and os.environ.get("SLURM_JOB_PARTITION") == "gpu"
             and os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1", "NoDevFiles"),
             "allocated GPU node required")
    _require(torch.cuda.is_available(), "allocated CUDA device unavailable")


def bind(path):
    return service._bind(Path(path))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return bind(path)


def load_request(binding, *, evaluation_binding=None):
    request = _json(binding)
    _require(request.get("schema_version") == "protocol_b_request_v1"
             and request.get("protocol") == "B" and type(request.get("group")) is int,
             "explicit Protocol B request required")
    observed = object_sha256(source_entries(ROOT))
    _require(request.get("source_sha256") == observed, "immutable scientific source differs from request")
    config = yaml.safe_load(_bytes(request["config"]))
    _require(isinstance(request.get("weights"), dict)
             and request["weights"].get("sha256") == service._FIXED["weights_sha256"],
             "official initialization binding required before query access")
    _bytes(request["weights"])
    context = bridge.load_reference_context(Path(request["records"]["path"]), Path(request["split"]["path"]),
        group=request["group"], records_sha256=request["records"]["sha256"],
        target_binding=request["target_enrollment"], enrollment_binding=request["enrollment"],
        raw_root=Path(request["raw_root"]), evaluation_binding=evaluation_binding)
    for name, expected in (("records", context.contract.records_sha256), ("split", context.contract.split_sha256)):
        _require(request[name]["sha256"] == expected, "request membership hash mismatch")
    _require(type(request.get("cache_bytes")) is int and 0 <= request["cache_bytes"] <= 8 * 1024**3,
             "bounded cache payload required")
    output = _path(request["output_root"])
    _require(not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output), "source/output overlap")
    service.effective_config(config, context, False)
    return request, context, config, output


def fresh_model(request, enabled, *, local_branch=True, reference_bn="shared"):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    configure_precision("cuda", "D_running_pilot")
    _bytes(request["weights"])
    return build_direct_reference_model(Path(request["weights"]["path"]), request["weights"]["sha256"],
                                        reference_enabled=enabled, head_seed=42, adapter_seed=42,
                                        local_enabled=local_branch, reference_bn=reference_bn).cuda()


def arm_model(request, config):
    recipe = dict(core.ARM_RECIPE_DEFAULTS, **{key: config[key] for key in core.ARM_RECIPE_DEFAULTS if key in config})
    return fresh_model(request, config["reference_enabled"], local_branch=recipe["local_branch"],
                       reference_bn=recipe["reference_bn"])


def checkpoint_ids(request, context, config, registration, phase_binding, model):
    metadata = json.loads(context.identities)
    return dict(source_sha256=request["source_sha256"], config_sha256=json_sha256(config),
        weights_sha256=request["weights"]["sha256"], records_sha256=metadata["records_sha256"],
        split_sha256=metadata["split_sha256"], enrollment_sha256=metadata["enrollment_manifest_sha256"],
        registration_sha256=registration["sha256"], phase_authority_sha256=phase_binding["sha256"],
        preprocess_sha256=json_sha256(dict(identities=metadata, policy=bridge._policy())),
        initialization_sha256=json_sha256(model.initialization_provenance))


def subset_loader(context, table, config, enabled, role, cache_bytes):
    data = bridge.ProtocolBReferenceDataset(context, table, role=role, reference_enabled=enabled,
        augment=dict(config["augment"], enabled=role == "train", seed=42), cache_bytes=cache_bytes)
    _require(len(data) >= 32, "smoke needs 32 source documents per role")
    return DataLoader(Subset(data, range(32)), batch_size=8 if role == "train" else 1,
                      shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(2043)), data


def invariant(model, loader):
    model.eval()
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 4:
            break
    joined = {}
    for name in ("query", "overlap", "context_available", "label", "reference"):
        if name in batches[0]:
            joined[name] = torch.cat([batch[name] for batch in batches])
    before = tensor_sha256(dict(model.named_buffers()))
    with torch.inference_mode():
        inputs, unused_labels = core._batch_inputs(model, joined, "cuda")
        grouped = core._forward(model, inputs)["attack_logit"]
        alone = []
        for batch in batches:
            inputs, unused_labels = core._batch_inputs(model, batch, "cuda")
            alone.append(core._forward(model, inputs)["attack_logit"])
        individual = torch.cat(alone)
    delta = (grouped - individual).abs()
    _require(torch.all(delta <= ATOL + RTOL * individual.abs()).item(), "batch companions changed document logits")
    _require(before == tensor_sha256(dict(model.named_buffers())), "invariance check changed stored statistics")
    return dict(passed=True, n_images=len(individual), max_abs_difference=float(delta.max()), buffers_unchanged=True)


def child_replay(binding, arm, phase, gate_sha256=None):
    arguments = [sys.executable, str(Path(__file__).resolve()), "replay", "--request", binding["path"],
                 "--request-sha256", binding["sha256"], "--arm", arm, "--phase", phase]
    if gate_sha256 is not None:
        arguments.extend(["--gate-sha256", gate_sha256])
    subprocess.run(arguments, check=True)


def smoke(binding):
    request, context, base, output = load_request(binding)
    _require(not (output / "gate").exists(), "smoke output already exists")
    table = bridge.build_protocol_b_registration(context, roles=["train", "val"], output_path=output / "registration.json")
    registration = bind(output / "registration.json")
    arm_reports = {}
    for arm, enabled in (("off", False), ("on", True)):
        config = service.effective_config(base, context, enabled)
        model = arm_model(request, config)
        initial_state = tensor_sha256(model.state_dict())
        identities = checkpoint_ids(request, context, config, registration, binding, model)
        optimizer = core.make_pilot_optimizer(model, {"track": "D_running_pilot"}, "train")
        scheduler = make_protocol_b_scheduler(optimizer, context.contract, config)
        train_loader, unused_train = subset_loader(context, table, config, enabled, "train", request["cache_bytes"])
        val_loader, unused_val = subset_loader(context, table, config, enabled, "val", 0)
        stats = core.train_direct_reference_epoch(model, train_loader, optimizer, scheduler, "cuda",
                                                  auxiliary_query_weight=config.get("auxiliary_query_weight", 0.))
        _require(stats["n_images"] == 32 and stats["optimizer_steps"] == 4, "incomplete smoke training")
        summary, predictions = core.evaluate_direct_reference(model, val_loader, "cuda")
        key = selection_key("accuracy", summary["accuracy"], summary["auc"], summary["image_loss"], stats["loss"])
        directory = output / "gate" / arm
        directory.mkdir(parents=True, exist_ok=False)
        core.save_direct_reference_checkpoint(directory / "selected.pt", model, config, identities, 1, key, "smoke")
        _write_csv(directory / "val_predictions.csv", predictions)
        report = dict(passed=True, no_test_open=True, config=config, identities=identities, stats=stats,
                      fresh_initial_state_sha256=initial_state, summary=summary, checkpoint=bind(directory / "selected.pt"),
                      predictions=bind(directory / "val_predictions.csv"), request=binding,
                      initialization=copy.deepcopy(model.initialization_provenance),
                      pid=os.getpid(), source_train_ids=context.rows("train").head(32).image_id.tolist(),
                      source_val_ids=context.rows("val").head(32).image_id.tolist(),
                      batch_invariance=invariant(model, val_loader))
        write(directory / "smoke.json", report)
        del model, optimizer, train_loader, unused_train, val_loader, unused_val
        torch.cuda.empty_cache()
        child_replay(binding, arm, "smoke")
        arm_reports[arm] = dict(smoke=bind(directory / "smoke.json"), replay=bind(directory / "fresh-replay.json"))
    metadata = json.loads(context.identities)
    gate = dict(schema_version="protocol_b_training_gate_v1", protocol="B", group=context.contract.group,
                source_sha256=request["source_sha256"], records_sha256=metadata["records_sha256"],
                split_sha256=metadata["split_sha256"], enrollment_sha256=metadata["enrollment_manifest_sha256"],
                registration_sha256=registration["sha256"], weights_sha256=request["weights"]["sha256"],
                recipe_sha256=json_sha256(service.effective_config(base, context, False)),
                passed=True, no_test_open=True, both_arm_smoke=True, replay=True, batch_invariance=True,
                request=binding, registration=registration, arm_reports=arm_reports,
                accounting=training_accounting(context.contract, service.effective_config(base, context, False)).__dict__)
    gate_binding = write(output / "gate/gate.json", gate)
    receipts.validate_training_gate(gate_binding, context, base, request["weights"], registration,
                                    request["source_sha256"], request_binding=binding)
    return gate_binding


def predictions(binding):
    content = _bytes(binding).decode()
    rows = list(csv.DictReader(content.splitlines()))
    for row in rows:
        row["label"] = int(row["label"])
        for name in ("attack_logit", "query_logit", "correction_logit", "attack_probability", "bonafide_score"):
            row[name] = float(row[name])
        for name in ("context_available", "referral_required"):
            _require(row[name] in ("True", "False"), "invalid prediction boolean")
            row[name] = row[name] == "True"
    return rows


def validate_prediction_rows(rows, context, phase):
    expected = context.rows("val")
    if phase == "smoke":
        expected = expected.head(32)
    service._predictions(dict(n_images=len(rows)), rows, expected)
    for row in rows:
        probability = float(torch.tensor(row["attack_logit"], dtype=torch.float32).sigmoid())
        _require(abs(row["attack_probability"] - probability) <= ATOL + RTOL * abs(probability)
                 and abs(row["bonafide_score"] - (1 - probability)) <= ATOL + RTOL * abs(1 - probability)
                 and row["context_available"] is not row["referral_required"], "prediction score/referral inconsistency")


def replay(binding, arm, phase, gate_sha256=None):
    request, context, base, output = load_request(binding)
    directory = output / ("gate" if phase == "smoke" else "training") / arm
    result = _json(bind(directory / ("smoke.json" if phase == "smoke" else "train.json")))
    if phase == "train":
        result, unused_config = receipts.validate_completed_training(context, request, base, arm, output,
                                pinned_gate(output, gate_sha256), binding, require_fresh=False)
    config = service.effective_config(base, context, arm == "on")
    _require(result.get("passed") is True and result["identities"]["source_sha256"] == request["source_sha256"], "unsafe replay result")
    registration = bind(output / "registration.json")
    table = bridge.load_protocol_b_registration(context, registration, roles=["train", "val"])
    model = arm_model(request, config)
    phase_binding = binding if phase == "smoke" else pinned_gate(output, gate_sha256)
    expected_ids = checkpoint_ids(request, context, config, registration, phase_binding, model)
    _require(result.get("identities") == expected_ids, "replay current source/data/initialization identities differ")
    _bytes(result["checkpoint"])
    canonical_checkpoint = directory / ("selected.pt" if phase == "smoke" else "best.pt")
    _require(result["checkpoint"] == bind(canonical_checkpoint), "noncanonical replay checkpoint")
    checkpoint = core.read_direct_reference_checkpoint(canonical_checkpoint, config, expected_ids)
    _require(checkpoint.get("phase") == phase, "checkpoint phase differs from replay request")
    if phase == "smoke":
        loader, unused_data = subset_loader(context, table, config, arm == "on", "val", 0)
    else:
        data = bridge.ProtocolBReferenceDataset(context, table, role="val", reference_enabled=arm == "on",
                                                augment=dict(config["augment"], enabled=False), cache_bytes=0)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    expected = predictions(result["predictions"])
    validate_prediction_rows(expected, context, phase)
    summary, actual, comparison = core.replay_direct_reference_checkpoint(Path(result["checkpoint"]["path"]), model,
        loader, "cuda", config=config, identities=result["identities"], expected_predictions=expected)
    _require(comparison["passed"] and len(actual) == (32 if phase == "smoke" else len(context.contract.source_val)), "fresh replay incomplete")
    validate_prediction_rows(actual, context, phase)
    from refer_or_clear.pilots import summarize_predictions
    metrics = summarize_predictions(expected)
    _require(all(summary.get(key) == value for key, value in metrics.items()), "replayed source metrics differ from parent predictions")
    return write(directory / "fresh-replay.json", dict(passed=True, fresh_process=True, pid=os.getpid(),
                 request=binding, checkpoint=result["checkpoint"], comparison=comparison, summary=summary,
                 identities=result["identities"], no_test_open=True, n_images=len(actual),
                 parent_result=bind(directory / ("smoke.json" if phase == "smoke" else "train.json"))))


def pinned_gate(output, expected_sha256):
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
             and all(character in "0123456789abcdef" for character in expected_sha256),
             "externally pinned successful diagnostic gate required")
    binding = dict(path=str(output / "gate/gate.json"), sha256=expected_sha256)
    _bytes(binding)
    return binding


def train(binding, arm, gate_sha256=None):
    request, context, base, output = load_request(binding)
    gate_binding = pinned_gate(output, gate_sha256)
    gate = _json(gate_binding)
    _require(gate.get("request") == binding, "training gate belongs to a different request")
    result = service.run_training(context, base, request["weights"], gate["registration"], output,
        reference_enabled=arm == "on", source_sha256=request["source_sha256"], gate_binding=gate_binding,
        device="cuda", cache_bytes=request["cache_bytes"])
    child_replay(binding, arm, "train", gate_sha256)
    return result


def completed_arm(request, context, base, output, arm, gate_binding=None, request_binding=None):
    directory = output / "training" / arm
    result = _json(bind(directory / "train.json"))
    config = service.effective_config(base, context, arm == "on")
    accounting = training_accounting(context.contract, config)
    _require(result.get("schema_version") == "protocol_b_train_v1"
             and result.get("passed") is True and result.get("no_test_open") is True
             and result.get("protocol") == "B" and type(result.get("group")) is int
             and result["group"] == context.contract.group and result.get("arm") == arm
             and type(result.get("epochs_completed")) is int and result["epochs_completed"] == 100
             and result.get("optimizer_steps") == accounting.total_optimizer_steps,
             "complete matched source training required before evaluation")
    _require(gate_binding is not None and request_binding is not None, "pinned training gate/request required")
    return receipts.validate_completed_training(context, request, base, arm, output, gate_binding, request_binding)


def evaluate(binding, gate_sha256=None):
    request, source_context, base, output = load_request(binding)
    _require(not (output / "evaluation").exists(), "locked target evaluation already exists")
    gate_binding = pinned_gate(output, gate_sha256)
    completed = {arm: completed_arm(request, source_context, base, output, arm, gate_binding, binding) for arm in ("off", "on")}
    authorization = dict(schema_version="protocol_b_evaluation_v1", **json.loads(source_context.identities),
                         source_validation_selected=True, authorized_image_ids=sorted(source_context.target.query_ids),
                         checkpoints={arm: result[0]["checkpoint"] for arm, result in completed.items()})
    evaluation_binding = write(output / "evaluation/authorization.json", authorization)
    request, context, base, output = load_request(binding, evaluation_binding=evaluation_binding)
    table = bridge.build_protocol_b_registration(context, roles=["test"], output_path=output / "evaluation/registration.json")
    results = {}
    for arm, (receipt, config) in completed.items():
        model = arm_model(request, config)
        checkpoint = core.read_direct_reference_checkpoint(Path(receipt["checkpoint"]["path"]), config, receipt["identities"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        data = bridge.ProtocolBReferenceDataset(context, table, role="test", reference_enabled=arm == "on",
                                               augment=dict(config["augment"], enabled=False), cache_bytes=0)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        summary, actual = core.evaluate_direct_reference(model, loader, "cuda")
        _require([row["image_id"] for row in actual] == sorted(context.target.query_ids), "target scoring population drift")
        directory = output / "evaluation" / arm
        directory.mkdir(exist_ok=False)
        _write_csv(directory / "test_predictions.csv", actual)
        results[arm] = write(directory / "test_summary.json", dict(summary, role="test", protocol="B", group=context.contract.group,
            selected_epoch=receipt["selected_epoch"], source_checkpoint=receipt["checkpoint"], authorization=evaluation_binding,
            registration=bind(output / "evaluation/registration.json"), published_target_count=len(context.contract.target_test),
            n_target_support=len(context.target.support_ids), n_target_excluded=len(context.target.excluded_ids),
            query_count=len(context.target.query_ids), published_test_population_preserved=False,
            threshold_policy="fixed_bonafide_threshold_0.5", predictions=bind(directory / "test_predictions.csv")))
        del model, checkpoint, data, loader
        torch.cuda.empty_cache()
    return write(output / "evaluation/result.json", dict(schema_version="protocol_b_evaluation_result_v1",
                 passed=True, protocol="B", group=context.contract.group, request=binding,
                 same_query_population=True, summaries=results, authorization=evaluation_binding))


def evaluate_reencoded(binding, gate_sha256=None, quality=75):
    """Score the frozen arms on queries re-encoded at one JPEG quality; native evaluation must exist first."""
    from refer_or_clear.data.reencode import reencoded_queries
    request, source_context, base, output = load_request(binding)
    native = _json(bind(output / "evaluation/result.json"))
    _require(native.get("passed") is True and native.get("request") == binding, "native locked evaluation required first")
    directory = output / f"evaluation-q{int(quality)}"
    _require(not directory.exists(), "re-encoded evaluation already exists")
    gate_binding = pinned_gate(output, gate_sha256)
    completed = {arm: completed_arm(request, source_context, base, output, arm, gate_binding, binding) for arm in ("off", "on")}
    evaluation_binding = bind(output / "evaluation/authorization.json")
    _require(native.get("authorization") == evaluation_binding, "native authorization binding mismatch")
    request, context, base, output = load_request(binding, evaluation_binding=evaluation_binding)
    registration = bind(output / "evaluation/registration.json")
    table = bridge.load_protocol_b_registration(context, registration, roles=["test"])
    results = {}
    with reencoded_queries(bridge, request["raw_root"], int(quality)) as treatment:
        for arm, (receipt, config) in completed.items():
            model = arm_model(request, config)
            checkpoint = core.read_direct_reference_checkpoint(Path(receipt["checkpoint"]["path"]), config, receipt["identities"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            data = bridge.ProtocolBReferenceDataset(context, table, role="test", reference_enabled=arm == "on",
                                 augment=dict(config["augment"], enabled=False), cache_bytes=0)
            loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
            summary, actual = core.evaluate_direct_reference(model, loader, "cuda")
            _require([row["image_id"] for row in actual] == sorted(context.target.query_ids), "target scoring population drift")
            arm_directory = directory / arm
            arm_directory.mkdir(parents=True, exist_ok=False)
            _write_csv(arm_directory / "test_predictions.csv", actual)
            results[arm] = write(arm_directory / "test_summary.json", dict(summary, role="test", protocol="B", group=context.contract.group,
                selected_epoch=receipt["selected_epoch"], source_checkpoint=receipt["checkpoint"], authorization=evaluation_binding,
                registration=registration, treatment=treatment, native_summary=native["summaries"][arm],
                threshold_policy="fixed_bonafide_threshold_0.5", predictions=bind(arm_directory / "test_predictions.csv")))
            del model, checkpoint, data, loader
            torch.cuda.empty_cache()
    return write(directory / "result.json", dict(schema_version="protocol_b_reencoded_evaluation_result_v1",
                 passed=True, protocol="B", group=context.contract.group, request=binding, treatment=treatment,
                 native=bind(output / "evaluation/result.json"), summaries=results, authorization=evaluation_binding))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "replay", "evaluate", "evaluate-reencoded"))
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--arm", choices=("off", "on"), default="off")
    parser.add_argument("--phase", choices=("smoke", "train"), default="train")
    parser.add_argument("--gate-sha256")
    args = parser.parse_args(argv)
    require_gpu_allocation()
    binding = dict(path=str(Path(args.request).resolve()), sha256=args.request_sha256)
    if args.command == "smoke":
        result = smoke(binding)
    elif args.command == "train":
        result = train(binding, args.arm, args.gate_sha256)
    elif args.command == "evaluate":
        result = evaluate(binding, args.gate_sha256)
    elif args.command == "evaluate-reencoded":
        result = evaluate_reencoded(binding, args.gate_sha256, args.quality)
    else:
        result = replay(binding, args.arm, args.phase, args.gate_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
