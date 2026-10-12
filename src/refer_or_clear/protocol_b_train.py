"""Matched Protocol B training service, not staging or allocation authorization.

The shared gate's recipe_sha256 is the canonical OFF effective-config hash.
Checkpoint config hashes remain arm-specific. A caller must authenticate source
externally and supply a hash-bound actual both-arm GPU gate. Only source train
and validation populations are consumed. Replay here is in-process.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import protocol_b_reference as bridge
from .data.direct_reference import _bytes, _json, _path, _require
from .data.protocol_b_reference import (
    ProtocolBReferenceDataset, ReferenceContext, load_protocol_b_registration,
)
from .direct_reference_pilot import (
    _validate_contract, _validate_initialization, _validate_model,
    evaluate_direct_reference, read_direct_reference_checkpoint,
    replay_direct_reference_checkpoint, save_direct_reference_checkpoint,
    train_direct_reference_epoch,
)
from .models.direct_reference import build_direct_reference_model
from .pilots import (
    OFFICIAL_WEIGHT_SHA256, _selection_record, _write_csv, configure_precision,
    file_sha256, json_sha256, make_pilot_optimizer, tensor_sha256,
)
from .protocol_b_contract import make_protocol_b_scheduler, training_accounting
from .protocol_b_receipts import validate_training_gate
from .train import selection_key


_FIXED = dict(
    schema_version="direct_reference_pilot_v1", track="direct_reference_pilot",
    architecture_id="anchor_direct_reference_ia18_v1", epochs=100, seed=42,
    head_seed=42, adapter_seed=42, batch_size=8, eval_batch_size=1,
    accumulation_steps=1, drop_last=True, eval_drop_last=False,
    input_shape=[4, 512, 800], mask_weight=0.0, clip_grad_norm=5.0,
    bn_policy="shared_sequential_train_bn_stored_eval_v1",
    initialization="fresh_offline_levir", wrong_design_fraction=0.0,
    comparison="matched_reference_bridge",
    weights_sha256=OFFICIAL_WEIGHT_SHA256["D_running_pilot"],
    optimizer=dict(name="adamw", lr_backbone=.0001, lr_new=.001, weight_decay=.05,
                   betas=[.9, .999], eps=1e-8),
    scheduler=dict(name="warmup_cosine", warmup_iters=1000, warmup_cap_fraction=.1),
    precision=dict(dtype="float32", amp=False, cudnn_tf32=False, matmul_tf32=False,
                   training_bit_reproducible=False),
    selection=dict(primary="accuracy", tie_break="auc", exact_tie="earlier_epoch"),
    normalization=dict(mean=[.485, .456, .406], std=[.229, .224, .225]),
    augment=dict(preset="direct_reference_canvas_v1", enabled=True, crop=None,
                 affine_px=8, rotation_degrees=1.0, jitter_px=2, jitter_p=.25,
                 photometric=True, photometric_strength=.2, reference_photometric=False,
                 jpeg_quality=None, interpolation="opencv_inter_linear",
                 binary_interpolation="opencv_inter_nearest", affine_border="constant_zero",
                 jitter_rgb_border="replicate", jitter_target_border="constant_zero", jitter_zone=True),
)


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value), "invalid SHA256 identity")
    return value


def effective_config(base, context, enabled):
    """Detach the exact model recipe and bind it to this B group and arm."""
    _require(isinstance(context, ReferenceContext) and type(enabled) is bool,
             "Protocol B context and boolean arm required")
    _require(isinstance(base, dict), "base config must be a mapping")
    permitted = set(_FIXED) | {"protocol", "group", "fold", "arm", "reference_enabled", "data_contract",
                              "records_csv", "split_csv", "records_sha256", "split_sha256", "enrollment_sha256"}
    _require(set(base) <= permitted, "unknown recipe fields or checkpoint/resume inputs")
    for key, value in _FIXED.items():
        _require(type(base.get(key)) is type(value) and json_sha256(base[key]) == json_sha256(value),
                 f"fixed Protocol B recipe mismatch: {key}")
    _require(not any(key in base for key in ("resume", "checkpoint", "warmstart", "init_checkpoint")),
             "fresh initialization forbids checkpoint/resume inputs")
    _require(base.get("protocol", "B") == "B" and base.get("group", context.contract.group) == context.contract.group,
             "foreign protocol/group config")
    _require(base.get("data_contract") in ("direct_reference_canvas_v1", "protocol_b_query_canvas_v1"),
             "unsupported data contract")
    metadata = json.loads(context.identities)
    _require(type(context.contract.group) is int and metadata.get("protocol") == "B" and type(metadata.get("group")) is int
             and metadata["group"] == context.contract.group and 0 <= metadata["group"] <= 9,
             "Protocol B context identity mismatch")
    for key, expected in (("records_sha256", context.contract.records_sha256),
                          ("split_sha256", context.contract.split_sha256)):
        _require(metadata.get(key) == _sha(expected), f"context {key} mismatch")
    bindings = context.bindings
    _require(len(bindings) >= 4 and bindings[0]["sha256"] == metadata["records_sha256"]
             and bindings[1]["sha256"] == metadata["split_sha256"]
             and bindings[2]["sha256"] == _sha(metadata.get("target_enrollment_sha256"))
             and bindings[3]["sha256"] == _sha(metadata.get("enrollment_manifest_sha256")),
             "context authorities mismatch")
    result = copy.deepcopy(base)
    result.pop("fold", None)
    result.update(protocol="B", group=context.contract.group, arm="on" if enabled else "off",
                  reference_enabled=enabled, data_contract="protocol_b_query_canvas_v1",
                  records_csv=bindings[0]["path"], split_csv=bindings[1]["path"],
                  records_sha256=metadata["records_sha256"], split_sha256=metadata["split_sha256"],
                  enrollment_sha256=metadata["enrollment_manifest_sha256"])
    training_accounting(context.contract, result)
    return result


def identities(checkpoint):
    """Return detached core identities from an already authenticated B checkpoint.

    This does not authenticate checkpoint file bytes or authorize execution.
    Callers must bind the file hash before using its contents for replay.
    """
    _require(isinstance(checkpoint, dict) and checkpoint.get("schema_version") == "direct_reference_pilot_v1",
             "direct-reference checkpoint required")
    config, value = checkpoint.get("config", {}), checkpoint.get("identities", {})
    _require(config.get("protocol") == "B" and type(config.get("group")) is int
             and 0 <= config["group"] <= 9 and "fold" not in config
             and config.get("data_contract") == "protocol_b_query_canvas_v1", "Protocol B checkpoint required")
    _validate_contract(config, value)
    _require(checkpoint.get("architecture_id") == config["architecture_id"]
             and checkpoint.get("arm") == config["arm"]
             and checkpoint.get("config_sha256") == json_sha256(config)
             and checkpoint.get("selected") is True and checkpoint.get("phase") == "train"
             and type(checkpoint.get("epoch")) is int and 1 <= checkpoint["epoch"] <= 100,
             "selected Protocol B training checkpoint mismatch")
    _validate_initialization(checkpoint.get("initialization_provenance", {}), config, value)
    return copy.deepcopy(value)


def _rows(context):
    rows = {role: context.rows(role) for role in ("train", "val")}
    for role, members in (("train", context.contract.source_train), ("val", context.contract.source_val)):
        expected = sorted(members, key=lambda row: row.image_id)
        frame = rows[role]
        _require(set(("image_id", "role", "design_id", "label", "attack_family")) <= set(frame),
                 "missing source row metadata")
        _require(frame.image_id.tolist() == [row.image_id for row in expected]
                 and frame.image_id.is_unique and frame.role.eq(role).all(), "exact sorted source membership required")
        for row, member in zip(frame.to_dict("records"), expected):
            _require(row["design_id"] == member.design_id and type(row["label"]) is int
                     and row["label"] == member.label, "source row metadata mismatch")
            if role == "train":
                _require(row.get("identity_id") == member.identity_id
                         and row.get("component_id") == member.component_id, "training provenance mismatch")
    for first, second in ((context.contract.source_train, context.contract.source_val),
                          (context.contract.source_train + context.contract.source_val, context.contract.target_test)):
        for field in ("image_id", "identity_id", "component_id"):
            _require(not {getattr(row, field) for row in first} & {getattr(row, field) for row in second},
                     "source/validation/target membership overlaps")
    return rows


def _predictions(summary, predictions, rows):
    _require(type(summary.get("n_images")) is int and summary["n_images"] == len(rows)
             and len(predictions) == len(rows), "complete source validation required")
    _require([row["image_id"] for row in predictions] == rows.image_id.tolist(), "validation prediction order/membership mismatch")
    for prediction, row in zip(predictions, rows.to_dict("records")):
        _require(all(prediction.get(key) == row[key] for key in ("design_id", "label", "attack_family")),
                 "validation prediction metadata mismatch")


def _bind(path):
    return dict(path=str(path), sha256=file_sha256(path))


def _write(path, document):
    with path.open("x") as stream:
        stream.write(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path.chmod(0o444)
    return _bind(path)


def run_training(context, config, weights_binding, registration_binding, output_root, *,
                 reference_enabled, source_sha256, gate_binding, device="cuda", cache_bytes=4 * 1024**3):
    """Run fixed 100-epoch fresh B training and selected validation replay.

    The retained-payload cache budget is shared proportionally by train/val
    population, not duplicated per dataset. It does not bound process RSS.
    """
    effective = effective_config(config, context, reference_enabled)
    accounting = training_accounting(context.contract, effective)
    _require(type(cache_bytes) is int and cache_bytes >= 0, "nonnegative integer cache budget required")
    metadata = json.loads(context.identities)
    gate = validate_training_gate(gate_binding, context, config, weights_binding,
                                  registration_binding, source_sha256)
    rows = _rows(context)
    root = _path(str(Path(output_root).absolute()))
    _require(root == _path(_json(gate["request"])["output_root"]), "training output differs from bound request")
    directory = _path(str(root / "training" / effective["arm"]))
    _require(not directory.exists(), "training arm output already exists")
    protected = [context.raw_root, Path(context.manifest["output_root"])]
    protected.extend(Path(binding["path"]) for binding in
                     [*context.bindings, weights_binding, registration_binding, gate_binding])
    for path in protected:
        path = _path(str(path))
        _require(not directory.is_relative_to(path) and not path.is_relative_to(directory), "output overlaps protected input")
    for binding in context.bindings:
        _bytes(binding)
    _bytes(weights_binding)
    table = load_protocol_b_registration(context, registration_binding, roles=["train", "val"])
    directory.mkdir(parents=True, exist_ok=False)

    def fresh_model():
        random.seed(effective["seed"])
        np.random.seed(effective["seed"])
        torch.manual_seed(effective["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective["seed"])
        configure_precision(device, "D_running_pilot")
        return build_direct_reference_model(Path(weights_binding["path"]), weights_binding["sha256"],
                    reference_enabled=reference_enabled, head_seed=effective["head_seed"],
                    adapter_seed=effective["adapter_seed"]).to(device)

    model = fresh_model()
    checkpoint_ids = dict(source_sha256=source_sha256, config_sha256=json_sha256(effective),
        weights_sha256=weights_binding["sha256"], records_sha256=metadata["records_sha256"],
        split_sha256=metadata["split_sha256"], enrollment_sha256=metadata["enrollment_manifest_sha256"],
        registration_sha256=registration_binding["sha256"], phase_authority_sha256=gate_binding["sha256"],
        preprocess_sha256=json_sha256(dict(identities=metadata, policy=bridge._policy())),
        initialization_sha256=json_sha256(model.initialization_provenance))
    _validate_contract(effective, checkpoint_ids)
    _validate_model(model, effective, checkpoint_ids)
    fresh_state = tensor_sha256(model.state_dict())
    smoke = _json(gate["arm_reports"][effective["arm"]]["smoke"])
    _require(fresh_state == smoke["fresh_initial_state_sha256"]
             and model.initialization_provenance == smoke["initialization"],
             "fresh training initialization differs from authenticated smoke")
    optimizer = make_pilot_optimizer(model, {"track": "D_running_pilot"}, "train")
    _require(not optimizer.state, "fresh optimizer state required")
    scheduler = make_protocol_b_scheduler(optimizer, context.contract, effective)
    val_cache_bytes = cache_bytes * accounting.n_val // (accounting.n_train + accounting.n_val)
    train_cache_bytes = cache_bytes - val_cache_bytes
    train_dataset = ProtocolBReferenceDataset(context, table, role="train", reference_enabled=reference_enabled,
                     augment=dict(effective["augment"], seed=effective["seed"]), cache_bytes=train_cache_bytes)
    val_dataset = ProtocolBReferenceDataset(context, table, role="val", reference_enabled=reference_enabled,
                   augment=dict(effective["augment"], enabled=False, seed=effective["seed"]), cache_bytes=val_cache_bytes)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(effective["seed"] + 3001))
    sampler = torch.Generator().manual_seed(effective["seed"] + 1001)
    history, omissions = [], []
    best_key, best_epoch, selected_predictions = None, 0, None
    for epoch in range(1, 101):
        order = torch.randperm(accounting.n_train, generator=sampler).tolist()
        omissions.append(dict(epoch=epoch, image_ids=rows["train"].iloc[order[accounting.images_per_epoch:]].image_id.tolist()))
        loader = DataLoader(train_dataset, batch_size=8, sampler=order, drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(effective["seed"] + 2001 + epoch))
        stats = train_direct_reference_epoch(model, loader, optimizer, scheduler, device)
        _require(type(stats.get("n_images")) is int and stats["n_images"] == accounting.images_per_epoch
                 and type(stats.get("optimizer_steps")) is int and stats["optimizer_steps"] == accounting.batches_per_epoch,
                 "full epoch update/drop-last mismatch")
        summary, predictions = evaluate_direct_reference(model, val_loader, device)
        _predictions(summary, predictions, rows["val"])
        key = selection_key("accuracy", summary["accuracy"], summary["auc"], summary["image_loss"], stats["loss"])
        history.append(dict(epoch=epoch, **stats, val_accuracy=summary["accuracy"], val_auc=summary["auc"],
            val_image_loss=summary["image_loss"], lr_pretrained=optimizer.param_groups[0]["lr"], lr_new=optimizer.param_groups[-1]["lr"]))
        _write_csv(directory / "log.csv", history)
        save_direct_reference_checkpoint(directory / "last.pt", model, effective, checkpoint_ids, epoch, key, "train")
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            selected_predictions = copy.deepcopy(predictions)
            save_direct_reference_checkpoint(directory / "best.pt", model, effective, checkpoint_ids, epoch, key, "train")
    _require(scheduler.last_epoch == accounting.total_optimizer_steps, "scheduler update accounting mismatch")
    checkpoint = read_direct_reference_checkpoint(directory / "best.pt", effective, checkpoint_ids)
    _require(identities(checkpoint) == checkpoint_ids and checkpoint["epoch"] == best_epoch
             and checkpoint["selection_key"] == _selection_record(best_key), "selected checkpoint mismatch")
    replay_model = fresh_model()
    _validate_model(replay_model, effective, checkpoint_ids)
    _require(tensor_sha256(replay_model.state_dict()) == fresh_state, "replay initialization must be fresh")
    summary, predictions, replay = replay_direct_reference_checkpoint(directory / "best.pt", replay_model, val_loader, device,
        config=effective, identities=checkpoint_ids, expected_predictions=selected_predictions)
    _predictions(summary, predictions, rows["val"])
    _require(replay.get("passed") is True, "selected validation replay failed")
    replay = dict(replay, fresh_process=False, n_images=accounting.n_val, phase="train")
    _write_csv(directory / "val_predictions.csv", predictions)
    _write(directory / "val_summary.json", dict(summary, role="val", protocol="B", group=context.contract.group, selected_epoch=best_epoch))
    _write(directory / "replay.json", replay)
    receipt = dict(schema_version="protocol_b_train_v1", protocol="B", group=context.contract.group, pid=os.getpid(),
        group_name=context.contract.group_name, passed=True, no_test_open=True, phase="train", diagnostic_only=False,
        arm=effective["arm"], reference_enabled=reference_enabled, role="val", epochs_completed=100,
        selected_epoch=best_epoch, optimizer_steps=accounting.total_optimizer_steps, n_train=accounting.n_train,
        n_val=accounting.n_val, batches_per_epoch=accounting.batches_per_epoch, images_per_epoch=accounting.images_per_epoch,
        scheduler=dict(total_updates=accounting.total_optimizer_steps, warmup_updates=accounting.effective_warmup_iters),
        cache_bytes=dict(total=cache_bytes, train=train_cache_bytes, val=val_cache_bytes),
        fresh_initial_state_sha256=fresh_state, initialization=copy.deepcopy(model.initialization_provenance),
        identities=checkpoint_ids, selection_key=_selection_record(best_key), omitted_train_ids=omissions, replay=replay,
        checkpoint=_bind(directory / "best.pt"), predictions=_bind(directory / "val_predictions.csv"),
        registration=copy.deepcopy(registration_binding), diagnostic_gate=copy.deepcopy(gate_binding),
        request=copy.deepcopy(gate["request"]), history=_bind(directory / "log.csv"),
        summary=_bind(directory / "val_summary.json"), in_process_replay=_bind(directory / "replay.json"))
    _write(directory / "train.json", receipt)
    return receipt
