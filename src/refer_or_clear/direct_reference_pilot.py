"""Classification-only direct-reference execution and selected-checkpoint replay."""

from pathlib import Path

import torch
from torch import nn

from .pilots import (
    OFFICIAL_WEIGHT_SHA256,
    _image_loss,
    _selection_record,
    compare_replay,
    json_sha256,
    make_pilot_optimizer,
    pilot_scheduler,
    select_phase_rows,
    summarize_predictions,
    tensor_sha256,
)
from .train import selection_key

SCHEMA_VERSION = "direct_reference_pilot_v1"
ARCHITECTURE_ID = "anchor_direct_reference_ia18_v1"
RECIPE_TRACK = "D_running_pilot"
IDENTITY_FIELDS = (
    "source_sha256", "config_sha256", "weights_sha256", "records_sha256", "split_sha256",
    "enrollment_sha256", "registration_sha256", "phase_authority_sha256", "preprocess_sha256",
    "initialization_sha256",
)


ARM_RECIPE_DEFAULTS = dict(local_branch=True, auxiliary_query_weight=0., reference_bn="shared")


def arm_recipe(base, enabled):
    """Resolve the per-arm recipe (local branch, auxiliary query BCE, reference BN policy).

    ``base["arm_recipe"]`` may hold ``{"off": {...}, "on": {...}}`` overrides of
    ``ARM_RECIPE_DEFAULTS``; absent keys keep the original DirectReference recipe.
    """
    if type(enabled) is not bool:
        raise ValueError("arm must be a boolean")
    recipes = base.get("arm_recipe")
    if recipes is None:
        recipes = {}
    if isinstance(recipes, dict):
        # YAML 1.1 reads unquoted ``off``/``on`` keys as booleans; accept both spellings.
        recipes = {{False: "off", True: "on"}.get(key, key) if type(key) is bool else key: value
                   for key, value in recipes.items()}
    if not isinstance(recipes, dict) or not set(recipes) <= {"off", "on"}:
        raise ValueError("arm_recipe must map 'off'/'on' to recipe overrides")
    resolved = {}
    for arm in ("off", "on"):
        overrides = recipes.get(arm, {})
        if not isinstance(overrides, dict) or not set(overrides) <= set(ARM_RECIPE_DEFAULTS):
            raise ValueError(f"unknown arm_recipe fields for {arm}")
        values = dict(ARM_RECIPE_DEFAULTS, **overrides)
        if type(values["local_branch"]) is not bool:
            raise ValueError("local_branch must be boolean")
        weight = values["auxiliary_query_weight"]
        if type(weight) not in (int, float) or not 0. <= float(weight) <= 10.:
            raise ValueError("auxiliary_query_weight must be a number in [0, 10]")
        values["auxiliary_query_weight"] = float(weight)
        if values["reference_bn"] not in ("shared", "stored"):
            raise ValueError("reference_bn must be 'shared' or 'stored'")
        if arm == "on" and not values["local_branch"]:
            raise ValueError("the reference-enabled arm requires local_branch")
        resolved[arm] = values
    return resolved["on" if enabled else "off"]


def _strict_fp32(model, device):
    device = torch.device(device)
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError("strict FP32 execution rejects autocast")
    if device.type == "cuda" and (torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError("strict FP32 execution rejects TF32")
    if type(getattr(model, "reference_enabled", None)) is not bool:
        raise ValueError("model must declare reference_enabled")
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("strict FP32 model tensors required")


def _batch_inputs(model, batch, device):
    query = batch["query"]
    overlap = batch["overlap"]
    available = batch["context_available"]
    labels = batch["label"]
    if query.dtype != torch.float32 or overlap.dtype != torch.float32:
        raise ValueError("strict FP32 query and overlap required")
    if query.ndim != 4 or query.shape[1] != 4 or overlap.shape != (len(query), 1, *query.shape[2:]):
        raise ValueError("invalid query/overlap shapes")
    if available.dtype != torch.bool or available.shape != (len(query),) or labels.shape != (len(query),):
        raise ValueError("invalid availability/label shapes")
    if not torch.isfinite(query).all() or not torch.isfinite(overlap).all() or not ((overlap == 0) | (overlap == 1)).all():
        raise ValueError("non-finite query or nonbinary overlap")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("classification labels must be binary")
    reference = None
    if model.reference_enabled:
        reference = batch.get("reference")
        if reference is not None:
            if not isinstance(reference, torch.Tensor) or reference.dtype != torch.float32:
                raise ValueError("strict FP32 reference required")
            if reference.shape != (len(query), 3, *query.shape[2:]):
                raise ValueError("invalid reference shape")
            if not torch.isfinite(reference[available]).all():
                raise ValueError("non-finite available reference")
            reference = reference.to(device)
        elif available.any():
            raise ValueError("available on-mode samples require reference")
    return {"query": query.to(device), "overlap": overlap.to(device), "reference": reference,
            "context_available": available.to(device)}, labels.to(device)


def _forward(model, inputs):
    outputs = model(inputs["query"], overlap=inputs["overlap"], reference=inputs["reference"],
                    context_available=inputs["context_available"])
    for key in ("attack_logit", "query_logit", "correction_logit"):
        value = outputs[key]
        if value.dtype != torch.float32:
            raise ValueError("strict FP32 logits required")
        if value.shape != (len(inputs["query"]),) or not torch.isfinite(value).all():
            raise ValueError("non-finite or invalid classification logits")
    return outputs


def auxiliary_query_loss(outputs, labels, weight):
    """Weighted BCE on the global query logit alone; zero weight contributes nothing."""
    if type(weight) not in (int, float) or isinstance(weight, bool) or not 0. <= float(weight) <= 10.:
        raise ValueError("auxiliary_query_weight must be a number in [0, 10]")
    if float(weight) == 0.:
        return None
    return float(weight) * _image_loss({"attack_logit": outputs["query_logit"]}, labels, track=RECIPE_TRACK)


def train_direct_reference_epoch(model, loader, optimizer, scheduler, device, *, auxiliary_query_weight=0.):
    """One epoch; ``loss`` is the fused BCE plus the weighted auxiliary query BCE."""
    _strict_fp32(model, device)
    auxiliary_query_loss({"query_logit": torch.zeros(1)}, torch.zeros(1), auxiliary_query_weight)
    model.train()
    loss_sum, image_sum, aux_sum, n_images, updates = 0., 0., 0., 0, 0
    for batch in loader:
        inputs, labels = _batch_inputs(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, inputs)
        image_loss = _image_loss(outputs, labels, track=RECIPE_TRACK)
        auxiliary = auxiliary_query_loss(outputs, labels, auxiliary_query_weight)
        loss = image_loss if auxiliary is None else image_loss + auxiliary
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError("non-finite training gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * len(labels)
        image_sum += image_loss.item() * len(labels)
        aux_sum += (0. if auxiliary is None else auxiliary.item()) * len(labels)
        n_images += len(labels)
        updates += 1
    if not n_images:
        raise ValueError("training loader has no images")
    return {"loss": loss_sum / n_images, "image_loss": image_sum / n_images, "aux_query_loss": aux_sum / n_images,
            "bce_map": None, "dice_map": None, "n_images": n_images, "optimizer_steps": updates}


def evaluate_direct_reference(model, loader, device):
    _strict_fp32(model, device)
    model.eval()
    before = tensor_sha256(dict(model.named_buffers()))
    predictions, loss_sum = [], 0.
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = _batch_inputs(model, batch, device)
            outputs = _forward(model, inputs)
            loss = _image_loss(outputs, labels, track=RECIPE_TRACK, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("non-finite evaluation loss")
            loss_sum += loss.item()
            # Float64 and sigmoid(-logit) for the bona fide score: a float32 sigmoid saturates above ~17 logits and
            # 1 - p cancels, which ties large logits and understates the AUC. Ranking must follow the logit.
            attack_logits = outputs["attack_logit"].detach().double()
            probabilities = attack_logits.sigmoid().cpu().tolist()
            bonafide = (-attack_logits).sigmoid().cpu().tolist()
            logits = {key: outputs[key].cpu().tolist() for key in ("attack_logit", "query_logit", "correction_logit")}
            for index, probability in enumerate(probabilities):
                meta = {key: str(batch["meta"][key][index]) for key in ("image_id", "design_id", "attack_family")}
                available = bool(inputs["context_available"][index].item())
                row = dict(meta, label=int(labels[index].item()), attack_probability=probability,
                           bonafide_score=bonafide[index], context_available=available, referral_required=not available)
                row.update({key: logits[key][index] for key in ("attack_logit", "query_logit", "correction_logit")})
                if "availability_reason" in batch["meta"]:
                    row["availability_reason"] = str(batch["meta"]["availability_reason"][index])
                predictions.append(row)
    if before != tensor_sha256(dict(model.named_buffers())):
        raise ValueError("evaluation changed model buffers")
    if not predictions:
        raise ValueError("evaluation loader has no images")
    if len({row["image_id"] for row in predictions}) != len(predictions):
        raise ValueError("duplicate evaluation image IDs")
    summary = summarize_predictions(predictions)
    n_available = sum(row["context_available"] for row in predictions)
    summary.update(image_loss=loss_sum / len(predictions), buffers_unchanged=True, n_context_available=n_available,
                   n_context_unavailable=len(predictions) - n_available,
                   referral_ids=[row["image_id"] for row in predictions if row["referral_required"]])
    return summary, predictions


def _validate_contract(config, identities):
    if config.get("schema_version") != SCHEMA_VERSION or config.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("direct-reference schema or architecture mismatch")
    enabled = config.get("reference_enabled")
    if type(enabled) is not bool or config.get("arm") != ("on" if enabled else "off"):
        raise ValueError("direct-reference arm mismatch")
    for key in IDENTITY_FIELDS:
        value = identities.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"missing or invalid identity {key}")
    if identities["config_sha256"] != json_sha256(config):
        raise ValueError("effective config identity mismatch")
    if identities["weights_sha256"] != OFFICIAL_WEIGHT_SHA256[RECIPE_TRACK]:
        raise ValueError("original official offline initialization identity mismatch")


def _validate_initialization(provenance, config, identities):
    if json_sha256(provenance) != identities["initialization_sha256"]:
        raise ValueError("initialization provenance identity mismatch")
    if provenance.get("architecture_id") != ARCHITECTURE_ID or provenance.get("reference_enabled") is not config["reference_enabled"]:
        raise ValueError("initialization architecture or arm mismatch")
    base = provenance.get("base", {})
    if base.get("requested_initialization") != "levir" or base.get("checkpoint_sha256") != identities["weights_sha256"]:
        raise ValueError("initialization must be fresh declared offline LEVIR")


def _validate_model(model, config, identities):
    if model.architecture_id != ARCHITECTURE_ID or model.reference_enabled is not config["reference_enabled"]:
        raise ValueError("model architecture or arm mismatch")
    _validate_initialization(model.initialization_provenance, config, identities)


def save_direct_reference_checkpoint(path, model, config, identities, epoch, val_key, phase):
    _validate_contract(config, identities)
    _validate_model(model, config, identities)
    if type(epoch) is not int or epoch < 1 or phase not in ("smoke", "train"):
        raise ValueError("selected checkpoint epoch or phase mismatch")
    checkpoint = {"schema_version": SCHEMA_VERSION, "architecture_id": ARCHITECTURE_ID, "arm": config["arm"],
                  "config": config, "config_sha256": json_sha256(config), "identities": identities,
                  "epoch": epoch, "selection_key": _selection_record(val_key), "phase": phase, "selected": True,
                  "state_dict": model.state_dict(), "initialization_provenance": model.initialization_provenance}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def read_direct_reference_checkpoint(path, config, identities):
    _validate_contract(config, identities)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID or checkpoint.get("arm") != config["arm"]:
        raise ValueError("checkpoint architecture or arm mismatch")
    if checkpoint.get("config") != config or checkpoint.get("config_sha256") != json_sha256(config):
        raise ValueError("checkpoint effective config mismatch")
    if checkpoint.get("identities") != identities:
        raise ValueError("checkpoint source/data/phase identities mismatch")
    if checkpoint.get("selected") is not True or checkpoint.get("phase") not in ("smoke", "train"):
        raise ValueError("replay requires a selected smoke/train checkpoint")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] < 1:
        raise ValueError("checkpoint epoch mismatch")
    _validate_initialization(checkpoint.get("initialization_provenance", {}), config, identities)
    return checkpoint


def replay_direct_reference_checkpoint(path, model, loader, device, *, config, identities, expected_predictions):
    checkpoint = read_direct_reference_checkpoint(path, config, identities)
    _validate_model(model, config, identities)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    summary, predictions = evaluate_direct_reference(model, loader, device)
    replay = compare_replay(expected_predictions, predictions)
    if not replay["passed"]:
        raise ValueError(f"selected checkpoint replay failed: {replay['failure']}")
    return summary, predictions, replay
