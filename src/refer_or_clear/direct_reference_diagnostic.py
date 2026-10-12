"""Bounded direct-reference diagnostics with explicit training-to-smoke gates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
import yaml

from .data import direct_reference as bridge
from .data.dataset import _normalize
from .data.query import load_role_rows
from .data.rectify import to_canvas
from .direct_reference_pilot import (
    evaluate_direct_reference, read_direct_reference_checkpoint, replay_direct_reference_checkpoint,
    save_direct_reference_checkpoint, train_direct_reference_epoch,
)
from .models.direct_reference import build_direct_reference_model
from .pilots import (
    _bn_snapshot, _write_csv, configure_precision, file_sha256, json_sha256,
    make_pilot_optimizer, optimizer_coverage, pilot_scheduler, select_phase_rows, tensor_sha256,
)
from .train import selection_key


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def write_receipt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o444)
    return bind(path)


def run_diagnostic_stages(context, training_stage, smoke_stage):
    training, smoke = {}, {}
    for arm, enabled in (("off", False), ("on", True)):
        receipt = training_stage(enabled)
        bridge._gate(receipt, context["identities"], context["weights_sha256"], enabled)
        training[arm] = receipt
    for arm, enabled in (("off", False), ("on", True)):
        bridge._gate(training[arm], context["identities"], context["weights_sha256"], enabled)
    for arm, enabled in (("off", False), ("on", True)):
        receipt = smoke_stage(enabled, copy.deepcopy(training))
        bridge._gate(receipt, context["identities"], context["weights_sha256"], enabled, smoke=True)
        smoke[arm] = receipt
    for receipts, is_smoke in ((training, False), (smoke, True)):
        for arm, enabled in (("off", False), ("on", True)):
            bridge._gate(receipts[arm], context["identities"], context["weights_sha256"], enabled, is_smoke)
    return {"passed": True, "training_gates": training, "smoke_gates": smoke}


def _staging_module():
    path = Path(__file__).resolve().parents[2] / "scripts/direct_reference_job.py"
    spec = importlib.util.spec_from_file_location("_direct_reference_diagnostic_staging", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_context(request_path, request_sha256, run_dir, device):
    require(str(device).startswith("cuda") and torch.cuda.is_available(), "allocated CUDA diagnostic required")
    staging = _staging_module()
    staging.validate_allocation()
    request = staging.read_json(staging.authenticate({"path": str(request_path), "sha256": request_sha256}))
    run_dir = Path(run_dir).resolve(strict=True)
    manifest = staging.read_json(run_dir / "input_manifest.json")
    require(manifest.get("schema_version") == "direct_reference_job_v1" and manifest.get("diagnostic_only") is True,
            "bounded staged diagnostic manifest required")
    require(manifest["request"] == {"path": str(request_path), "sha256": request_sha256}, "request binding mismatch")
    source = Path(manifest["paths"]["source_dir"])
    require(source == Path(__file__).resolve().parents[2], "imported scientific source mismatch")
    require(staging.source_digest(source) == request["source_sha256"] == manifest["source_sha256"], "source byte mismatch")
    for name, authority in manifest["authorities"].items():
        staging.authenticate(authority)
        require(authority["sha256"] == request[name]["sha256"], f"frozen {name} authority mismatch")
    require(manifest["membership_counts"] == {"train": 1778, "val": 222}, "official source membership required")
    config = yaml.safe_load(Path(manifest["authorities"]["config"]["path"]).read_text())
    reference_config = yaml.safe_load((source / "configs/direct_reference_pilot.yaml").read_text())
    require(config == reference_config, "diagnostic recipe must equal the reviewed source configuration")
    require(config["weights_sha256"] == bridge.WEIGHTS_SHA256 and config["enrollment_sha256"] == bridge.ENROLLMENT_SHA256,
            "reviewed original weights and enrollment required")
    identities = {"source_sha256": manifest["source_sha256"], "recipe_sha256": json_sha256(config),
                  "records_sha256": manifest["authorities"]["records"]["sha256"],
                  "split_sha256": manifest["authorities"]["split"]["sha256"],
                  "enrollment_manifest_sha256": manifest["authorities"]["enrollment"]["sha256"]}
    return {"manifest": manifest, "config": config, "identities": identities, "weights_sha256": config["weights_sha256"],
            "run_dir": run_dir, "device": torch.device(device), "request": bind(Path(request_path))}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(dataset, batch_size, *, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(1043), drop_last=False)


def _model(context, enabled):
    _seed(context["config"]["seed"])
    configure_precision(context["device"], "D_running_pilot")
    weights = context["manifest"]["authorities"]["weights"]
    model = build_direct_reference_model(Path(weights["path"]), weights["sha256"], reference_enabled=enabled,
                                         head_seed=context["config"]["head_seed"], adapter_seed=context["config"]["adapter_seed"])
    optimizer_coverage(model)
    return model.to(context["device"])


def _initialization_receipt(model):
    provenance = copy.deepcopy(model.initialization_provenance)
    return {"state_sha256": tensor_sha256(model.state_dict()),
            "initialization_sha256": json_sha256(provenance), "provenance": provenance}


@contextmanager
def pixel_audit(context, allowed):
    original = bridge._image
    raw_root = Path(context["manifest"]["paths"]["raw_root"])
    allowed_paths = {str(raw_root / row.path): (row.image_id, row.role) for row in allowed.itertuples()}
    opened = []

    def read(binding, flags):
        path = Path(binding["path"])
        if path.is_relative_to(raw_root):
            require(str(path) in allowed_paths, "query pixel access outside current phase")
            identifier, role = allowed_paths[str(path)]
            opened.append({"image_id": identifier, "role": role, "path": str(path)})
        return original(binding, flags)

    bridge._image = read
    try:
        yield opened
    finally:
        bridge._image = original


def _preprocess(context, authority):
    manifest = context["manifest"]
    return {"phase_authority": authority, "enrollment_authority": manifest["authorities"]["enrollment"],
            "product_root_authority": manifest["product_root_authority"],
            "records_authority": manifest["authorities"]["records"], "split_authority": manifest["authorities"]["split"],
            "raw_root": manifest["paths"]["raw_root"], "source_sha256": context["identities"]["source_sha256"],
            "recipe_sha256": context["identities"]["recipe_sha256"], "weights_sha256": context["weights_sha256"]}


def _checkpoint_identities(context, config, preprocess, table_binding, initialization_sha256):
    return {"source_sha256": context["identities"]["source_sha256"], "config_sha256": json_sha256(config),
            "weights_sha256": context["weights_sha256"], "records_sha256": context["identities"]["records_sha256"],
            "split_sha256": context["identities"]["split_sha256"],
            "enrollment_sha256": context["identities"]["enrollment_manifest_sha256"],
            "registration_sha256": table_binding["sha256"],
            "phase_authority_sha256": preprocess["phase_authority"]["sha256"],
            "preprocess_sha256": json_sha256(preprocess), "initialization_sha256": initialization_sha256}


def _prepare_table(context, rows, phase, training_gates=None):
    directory = context["run_dir"] / "diagnostic" / phase
    directory.mkdir(parents=True, exist_ok=True)
    phase_path = directory / "phase-authority.json"
    phase_payload = {"schema_version": "direct_reference_phase_v1", "phase": phase, **context["identities"],
                     "authorized_image_ids": {role: sorted(rows.loc[rows.role == role, "image_id"].tolist())
                                               for role in ("train", "val")}}
    if training_gates is not None:
        phase_payload["training_gates"] = training_gates
    authority = write_receipt(phase_path, phase_payload)
    preprocess = _preprocess(context, authority)
    enrollment = json.loads(Path(preprocess["enrollment_authority"]["path"]).read_text())
    with pixel_audit(context, rows) as opened:
        table = bridge.build_registration_table(rows, Path(preprocess["raw_root"]), enrollment, preprocess=preprocess)
    table_binding = write_receipt(directory / "registration.json", table)
    expected_reads = {(row.image_id, row.role) for row in rows.itertuples()}
    require(len(opened) == len(rows) and {(entry["image_id"], entry["role"]) for entry in opened} == expected_reads,
            "registration query read membership mismatch")
    ledger = write_receipt(directory / "registration-query-reads.json", {
        "schema_version": "direct_reference_query_reads_v1", "phase": phase, "shared": True,
        "identities": context["identities"], "phase_authority": authority,
        "registration": table_binding, "query_opens": opened})
    loaded = bridge.load_registration_table(Path(table_binding["path"]), table_binding["sha256"], rows=rows,
                                            enrollment_manifest=enrollment, preprocess=preprocess)
    return {"rows": rows, "directory": directory, "preprocess": preprocess, "enrollment": enrollment,
            "table": loaded, "table_binding": table_binding, "registration_pixel_opens": opened,
            "registration_query_reads": ledger}


def _dataset(context, prepared, rows, enabled, augmented=False):
    augment = copy.deepcopy(context["config"]["augment"]) if augmented else {"enabled": False}
    return bridge.DirectReferenceDataset(rows, Path(prepared["preprocess"]["raw_root"]), prepared["enrollment"],
                                         prepared["table"], reference_enabled=enabled,
                                         train=bool(rows.role.eq("train").all()), augment=augment)


def _invariance(model, dataset, device):
    model.eval()
    before = tensor_sha256(dict(model.named_buffers()))
    rows = [dataset[index] for index in range(min(8, len(dataset)))]
    require(len(rows) == 8, "eight inference probes required")

    def inputs(indices):
        return {"query": torch.stack([rows[index]["query"] for index in indices]).to(device),
                "overlap": torch.stack([rows[index]["overlap"] for index in indices]).to(device),
                "context_available": torch.stack([rows[index]["context_available"] for index in indices]).to(device),
                "reference": torch.stack([rows[index]["reference"] for index in indices]).to(device)
                             if model.reference_enabled else None}

    maximum = 0.
    with torch.inference_mode():
        singles = [model(**inputs([index]))["attack_logit"] for index in range(8)]
        for count in (1, 2, 4, 8):
            for target in range(8):
                for position in sorted({0, count // 2, count - 1}):
                    indices = [index for index in range(8) if index != target][:count - 1]
                    indices.insert(position, target)
                    value = model(**inputs(indices))["attack_logit"][position:position + 1]
                    torch.testing.assert_close(value, singles[target], atol=1e-5, rtol=1e-4)
                    maximum = max(maximum, float((value - singles[target]).abs().max().item()))
    require(before == tensor_sha256(dict(model.named_buffers())), "inference changed model buffers")
    return {"passed": True, "n_queries": 8, "batch_sizes": [1, 2, 4, 8], "max_abs_difference": maximum,
            "buffers_unchanged": True}


class Diagnostic:
    def __init__(self, context):
        self.context = context
        authorities = context["manifest"]["authorities"]
        records, split = Path(authorities["records"]["path"]), Path(authorities["split"]["path"])
        self.train_rows = load_role_rows(records, split, "train")
        self.prepared_training = None
        self.prepared_smoke = None

    def training(self, enabled):
        context = self.context
        probes = select_phase_rows(self.train_rows, "overfit", "train")
        coordinates = self.train_rows.loc[self.train_rows.label == 0].groupby("design_id", sort=True).head(1)
        require(coordinates.design_id.nunique() == 10, "training coordinate probes must cover all ten designs")
        union = pd.concat([coordinates, probes]).drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
        if self.prepared_training is None:
            self.prepared_training = _prepare_table(context, union, "coordinate")
        prepared = self.prepared_training
        arm = "on" if enabled else "off"
        directory = prepared["directory"] / arm
        directory.mkdir()
        with pixel_audit(context, union) as opened:
            coordinate_dataset = _dataset(context, prepared, coordinates, enabled)
            coordinate_details = []
            for index in range(len(coordinate_dataset)):
                item = coordinate_dataset[index]
                row = coordinate_dataset.rows[index]
                raw_path = Path(prepared["preprocess"]["raw_root"]) / row["path"]
                raw = bridge._image({"path": str(raw_path), "sha256": item["meta"]["query_sha256"]}, cv2.IMREAD_COLOR)
                canvas, _, _ = to_canvas(raw)
                require(torch.equal(item["query"][:3], torch.from_numpy(np.ascontiguousarray(_normalize(canvas)))),
                        "context bridge changed query pixels")
                require(item["context_available"].item(), "training genuine coordinate probe unavailable")
                coordinate_details.append({"image_id": item["meta"]["image_id"], "bank": item["meta"]["bank"],
                                           "available": True, "query_pixels_unchanged": True})
            model = _model(context, enabled)
            initial = _initialization_receipt(model)
            train_dataset = _dataset(context, prepared, probes, enabled)
            eval_dataset = _dataset(context, prepared, probes, enabled)
            optimizer = make_pilot_optimizer(model, {"track": "D_running_pilot"}, "overfit")
            require(not optimizer.state, "fresh overfit optimizer required")
            loader = _loader(train_dataset, 8, shuffle=True)
            updates = 0
            bn_audit = None
            while updates < 500:
                for batch in loader:
                    model.train()
                    arguments = {key: batch[key].to(context["device"]) for key in ("query", "overlap", "context_available")}
                    arguments["reference"] = batch["reference"].to(context["device"]) if enabled else None
                    before = _bn_snapshot(model)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(**arguments)
                    loss = functional.binary_cross_entropy_with_logits(output["attack_logit"], batch["label"].float().to(context["device"]))
                    require(torch.isfinite(loss).item(), "nonfinite overfit loss")
                    loss.backward()
                    require(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                                for parameter in model.parameters()), "nonfinite overfit gradient")
                    after = _bn_snapshot(model)
                    reference_call = enabled and arguments["context_available"].any().item()
                    increments = {name: after[name] - value for name, value in before.items()}
                    require(len(increments) == 22 and all(value == (2 if reference_call and name.startswith(
                        ("base.backbone.stem.", "base.backbone.layer1.", "base.backbone.layer2.")) else 1)
                        for name, value in increments.items()), "shared training BN counters mismatch")
                    bn_audit = {"passed": True, "increments": increments}
                    optimizer.step()
                    updates += 1
                    if updates >= 500:
                        break
                summary, predictions = evaluate_direct_reference(model, _loader(eval_dataset, 1), context["device"])
                correct = len(predictions) - len(summary["genuine_false_positives"]) - len(summary["altered_misses"])
                print(json.dumps({"stage": "overfit", "arm": arm, "updates": updates, "n_correct": correct,
                                  "image_loss": summary["image_loss"]}), flush=True)
                if correct >= 15 and summary["image_loss"] < .15:
                    break
            require(correct >= 15 and summary["image_loss"] < .15, "bounded overfit criterion failed")
            invariance = _invariance(model, eval_dataset, context["device"])
            _write_csv(directory / "predictions.csv", predictions)
        roles = sorted({entry["role"] for entry in opened})
        require(roles == ["train"], "training stage opened forbidden query role")
        receipt = {"schema_version": "direct_reference_training_gates_v1", **context["identities"],
                   "weights_sha256": context["weights_sha256"], "reference_enabled": enabled, "passed": True,
                   "no_test_open": True, "opened_roles": roles, "coordinate_gate": {"passed": True, "probes": coordinate_details},
                   "model_gate": {"passed": True, "bn_training": bn_audit, "inference": invariance,
                                  "actual_stack_report_sha256": file_sha256(context["run_dir"] / "actual-stack-test.xml")},
                   "overfit": {"n_images": 16, "n_correct": correct, "image_loss": summary["image_loss"], "updates": updates},
                   "initialization": initial, "registration": prepared["table_binding"], "pixel_opens": opened,
                   "pixel_opens_scope": "query_reads", "query_read_stage": "coordinate_overfit",
                   "registration_query_reads": prepared["registration_query_reads"]}
        del model, optimizer
        torch.cuda.empty_cache()
        return write_receipt(directory / "training-gate.json", receipt)

    def smoke(self, enabled, training_gates):
        context = self.context
        authorities = context["manifest"]["authorities"]
        if self.prepared_smoke is None:
            train = select_phase_rows(self.train_rows, "smoke", "train")
            val = select_phase_rows(load_role_rows(Path(authorities["records"]["path"]), Path(authorities["split"]["path"]), "val"), "smoke", "val")
            self.prepared_smoke = _prepare_table(context, pd.concat([train, val]).sort_values("image_id"), "smoke", training_gates)
        prepared = self.prepared_smoke
        arm = "on" if enabled else "off"
        directory = prepared["directory"] / arm
        directory.mkdir()
        config = copy.deepcopy(context["config"])
        config.update(arm=arm, reference_enabled=enabled)
        train_rows, val_rows = [prepared["rows"].loc[prepared["rows"].role == role].copy() for role in ("train", "val")]
        with pixel_audit(context, prepared["rows"]) as opened:
            model = _model(context, enabled)
            fresh = tensor_sha256(model.state_dict())
            training_receipt = bridge._json(training_gates[arm])
            require(fresh == training_receipt["initialization"]["state_sha256"], "smoke did not reset pristine initialization")
            require(json_sha256(model.initialization_provenance) == training_receipt["initialization"]["initialization_sha256"],
                    "smoke initialization provenance changed")
            train_dataset = _dataset(context, prepared, train_rows, enabled, augmented=True)
            val_dataset = _dataset(context, prepared, val_rows, enabled)
            optimizer = make_pilot_optimizer(model, {"track": "D_running_pilot"}, "smoke")
            require(not optimizer.state, "fresh smoke optimizer required")
            train_loader = _loader(train_dataset, 8, shuffle=True)
            scheduler = pilot_scheduler(optimizer, "D_running_pilot", len(train_loader))
            before_inference = _invariance(model, val_dataset, context["device"])
            history = train_direct_reference_epoch(model, train_loader, optimizer, scheduler, context["device"])
            summary, predictions = evaluate_direct_reference(model, _loader(val_dataset, 1), context["device"])
            require(history["n_images"] == 32 and summary["n_images"] == 32, "smoke did not cover declared membership")
            require([row["image_id"] for row in predictions] == val_rows.image_id.tolist(), "smoke prediction membership mismatch")
            after_inference = _invariance(model, val_dataset, context["device"])
            identities = _checkpoint_identities(context, config, prepared["preprocess"], prepared["table_binding"],
                                                json_sha256(model.initialization_provenance))
            checkpoint = directory / "best.pt"
            selected = selection_key("accuracy", summary["accuracy"], summary["auc"],
                                     summary["image_loss"], history["image_loss"])
            save_direct_reference_checkpoint(checkpoint, model, config, identities, 1, selected, "smoke")
            _write_csv(directory / "predictions.csv", predictions)
            replay_state = {"config": config, "identities": identities, "checkpoint": bind(checkpoint),
                            "predictions": bind(directory / "predictions.csv"), "table": prepared["table_binding"],
                            "preprocess": prepared["preprocess"], "val_ids": val_rows.image_id.tolist()}
            state_binding = write_receipt(directory / "replay-state.json", replay_state)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/check_direct_reference.py"),
                   "--request", context["request"]["path"], "--request-sha256", context["request"]["sha256"],
                   "--run-dir", str(context["run_dir"]), "--device", str(context["device"]), "--phase", "replay", "--arm", arm,
                   "--replay-state", state_binding["path"], "--replay-state-sha256", state_binding["sha256"]]
        with (directory / "replay.log").open("x") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        require(process.returncode == 0, f"fresh-process {arm} smoke replay failed")
        replay = json.loads((directory / "replay.json").read_text())
        require(replay.get("passed") is True and replay.get("fresh_process") is True, "smoke replay receipt failed")
        require(replay.get("replay_state") == state_binding and replay.get("checkpoint") == replay_state["checkpoint"]
                and replay.get("identities") == identities, "smoke replay artifact/identity mismatch")
        replay_count = replay.get("summary", {}).get("n_images")
        require(type(replay_count) is int and replay_count == 32 and replay.get("val_ids") == val_rows.image_id.tolist(),
                "smoke replay membership/order mismatch")
        require(isinstance(replay.get("pixel_opens"), list), "smoke replay query reads missing")
        bridge._bytes(state_binding)
        bridge._bytes(replay_state["checkpoint"])
        receipt = {"schema_version": "direct_reference_smoke_gates_v1", **context["identities"],
                   "weights_sha256": context["weights_sha256"], "reference_enabled": enabled, "passed": True,
                   "no_test_open": True, "n_images": {"train": 32, "val": 32}, "epochs_completed": 1,
                   "replay": replay, "training": history, "validation": summary, "pixel_opens": opened,
                   "initial_state_sha256": fresh, "checkpoint": bind(checkpoint),
                   "inference_before": before_inference, "inference_after": after_inference,
                   "registration": prepared["table_binding"], "registration_query_reads": prepared["registration_query_reads"],
                   "pixel_opens_scope": "query_reads", "query_read_stage": "smoke",
                   "replay_query_reads": replay["pixel_opens"]}
        return write_receipt(directory / "smoke-gate.json", receipt)


def replay(context, arm, *, state_binding):
    require(arm in ("off", "on"), "replay requires a pinned arm")
    directory = context["run_dir"] / "diagnostic/smoke" / arm
    require(state_binding.get("path") == str(directory / "replay-state.json"), "replay state path mismatch")
    state = bridge._json(state_binding)
    enabled = arm == "on"
    config = copy.deepcopy(context["config"])
    config.update(arm=arm, reference_enabled=enabled)
    require(state.get("config") == config, "replay configuration/arm mismatch")
    phase_binding = bind(directory.parent / "phase-authority.json")
    preprocess = _preprocess(context, phase_binding)
    require(state.get("preprocess") == preprocess, "replay preprocessing authority mismatch")
    phase = bridge._json(phase_binding)
    require(phase.get("schema_version") == "direct_reference_phase_v1" and phase.get("phase") == "smoke"
            and all(phase.get(key) == value for key, value in context["identities"].items()), "replay phase identity mismatch")
    authorities = context["manifest"]["authorities"]
    selected = {role: select_phase_rows(load_role_rows(Path(authorities["records"]["path"]),
                   Path(authorities["split"]["path"]), role), "smoke", role) for role in ("train", "val")}
    require(phase.get("authorized_image_ids") == {role: rows.image_id.tolist() for role, rows in selected.items()},
            "replay phase membership mismatch")
    gates = phase.get("training_gates", {})
    require(set(gates) == {"off", "on"}, "replay training gates missing")
    for gate_arm, active in (("off", False), ("on", True)):
        bridge._gate(gates[gate_arm], context["identities"], context["weights_sha256"], active)
    initialization = bridge._json(gates[arm])["initialization"]["initialization_sha256"]
    table_binding = bind(directory.parent / "registration.json")
    identities = _checkpoint_identities(context, config, preprocess, table_binding, initialization)
    require(state.get("identities") == identities, "replay checkpoint identities mismatch")
    expected_bindings = {"checkpoint": bind(directory / "best.pt"), "predictions": bind(directory / "predictions.csv"),
                         "table": table_binding}
    for key, binding in expected_bindings.items():
        require(state.get(key) == binding, f"replay {key} binding mismatch")
        bridge._bytes(binding)
    rows = selected["val"]
    require(len(rows) == 32 and rows.image_id.tolist() == state.get("val_ids"), "replay membership mismatch")
    read_direct_reference_checkpoint(Path(expected_bindings["checkpoint"]["path"]), config, identities)
    with Path(expected_bindings["predictions"]["path"]).open(newline="") as stream:
        expected = list(csv.DictReader(stream))
    require(len(expected) == 32 and [row["image_id"] for row in expected] == rows.image_id.tolist(),
            "replay prediction membership/order mismatch")
    enrollment = bridge._json(authorities["enrollment"])
    table = bridge.load_registration_table(Path(table_binding["path"]), table_binding["sha256"], rows=rows,
                                            enrollment_manifest=enrollment, preprocess=preprocess)
    prepared = {"table": table, "preprocess": preprocess, "enrollment": enrollment}
    with pixel_audit(context, rows) as opened:
        dataset = _dataset(context, prepared, rows, enabled)
        model = _model(context, enabled)
        summary, predictions, result = replay_direct_reference_checkpoint(Path(expected_bindings["checkpoint"]["path"]), model,
            _loader(dataset, 1), context["device"], config=config, identities=identities, expected_predictions=expected)
        invariance = _invariance(model, dataset, context["device"])
    require(type(summary["n_images"]) is int and summary["n_images"] == 32
            and [row["image_id"] for row in predictions] == rows.image_id.tolist(), "replay output membership mismatch")
    result.update(fresh_process=True, pid=os.getpid(), checkpoint=expected_bindings["checkpoint"], replay_state=state_binding,
                  identities=identities, val_ids=[row["image_id"] for row in predictions],
                  inference=invariance, summary=summary, pixel_opens=opened,
                  pixel_opens_scope="query_reads", query_read_stage="replay")
    write_receipt(directory / "replay.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "request-sha256", "run-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("diagnostic", "replay"), default="diagnostic")
    parser.add_argument("--arm", choices=("off", "on"))
    parser.add_argument("--replay-state")
    parser.add_argument("--replay-state-sha256")
    args = parser.parse_args(argv)
    if args.phase == "replay":
        require(args.arm is not None and args.replay_state is not None and args.replay_state_sha256 is not None,
                "replay requires pinned arm and external state binding")
    else:
        require(args.arm is None and args.replay_state is None and args.replay_state_sha256 is None,
                "diagnostic runs both arms without replay state")
    context = load_context(Path(args.request), args.request_sha256, args.run_dir, args.device)
    if args.phase == "replay":
        return replay(context, args.arm, state_binding={"path": args.replay_state, "sha256": args.replay_state_sha256})
    require(args.arm is None, "diagnostic runs both arms")
    staging = _staging_module()
    staging.check_stack_report(context["run_dir"] / "actual-stack-test.xml")
    diagnostic = Diagnostic(context)
    try:
        result = run_diagnostic_stages(context, diagnostic.training, diagnostic.smoke)
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as error:
        result = {"passed": False, "errors": [str(error)]}
    result.update(schema_version="direct_reference_diagnostic_v1", diagnostic_only=True,
                  identities=context["identities"], no_test_open=True)
    write_receipt(context["run_dir"] / "diagnostic-gate.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
