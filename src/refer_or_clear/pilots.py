"""Isolated, validation-only anchor training and diagnostic correctness gates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .anchor_contract import SCHEMA_VERSION, contract_for
from . import metrics as M
from .losses import referral_loss
from .train import make_scheduler, selection_key

ATOL = 1e-5
RTOL = 1e-4
ATTACK_SUBGROUPS = {"copy": "text_copy", "inpainting": "text_inpaint"}
OFFICIAL_WEIGHT_SHA256 = {
    "R": "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    "D": "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618",
    "N": "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_sha256(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def isolated_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _model_builder(track):
    if track == "R":
        from .models.anchor_rgb import build_anchor_rgb
        return build_anchor_rgb
    from .models.anchor_changer import build_anchor_changer
    return build_anchor_changer


def optimizer_coverage(model):
    groups = model.parameter_groups()
    if len(groups) != 2:
        raise ValueError("optimizer coverage requires pretrained and new parameter groups")
    active = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    supplied = [id(parameter) for group in groups for parameter in group]
    if len(set(supplied)) != len(supplied) or set(supplied) != set(active):
        raise ValueError("optimizer coverage contains duplicate, missing or inactive parameters")
    return {"passed": True, "n_tensors": len(active),
            "groups": [[active[id(parameter)] for parameter in group] for group in groups]}


def validate_weight_request(track, expected_sha256):
    contract_for(track)
    if expected_sha256.lower() != OFFICIAL_WEIGHT_SHA256[track]:
        raise ValueError("runtime requires the declared official offline initialization SHA256")


def validate_initialized_model(model, track, expected_sha256):
    contract = contract_for(track)
    if model.track != track or model.architecture_id != contract.architecture_id:
        raise ValueError("initialized model track or architecture mismatch")
    _validate_provenance(model.initialization_provenance, track, expected_sha256)
    return {"passed": True, "provenance": model.initialization_provenance, "optimizer_coverage": optimizer_coverage(model)}


def _validate_provenance(provenance, track, expected_sha256):
    requested = "IMAGENET1K_V1" if track == "R" else "levir"
    if provenance.get("requested_initialization") != requested:
        raise ValueError("requested initialization must be official offline pretrained weights")
    if provenance.get("checkpoint_sha256") != expected_sha256.lower():
        raise ValueError("initialization checkpoint hash mismatch")
    for key in ("checkpoint_path", "loaded_keys", "pretrained_tensor_sha256"):
        if not provenance.get(key):
            raise ValueError(f"initialization provenance missing {key}")
    tensor_hash = provenance["pretrained_tensor_sha256"]
    if len(tensor_hash) != 64 or any(character not in "0123456789abcdef" for character in tensor_hash):
        raise ValueError("invalid initialized tensor hash")
    if provenance.get("unexpected_keys") or "missing_keys" not in provenance:
        raise ValueError("initialization load-key audit is incomplete")
    if track == "R" and provenance["missing_keys"]:
        missing = set(provenance["missing_keys"])
        migrated = set(provenance.get("synthesized_keys", []))
        if (expected_sha256.lower() != "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
                or provenance.get("migration_policy") != "official_imagenet1k_v1_missing_bn_counters_zero_v1"
                or missing != migrated or not all(name.endswith(".num_batches_tracked") for name in missing)
                or provenance.get("synthesized_values") != {name: 0 for name in missing}):
            raise ValueError("ImageNet initialization has unallowlisted missing keys or migration")


def build_initialized_model(config, weights, weights_sha256):
    if len(weights_sha256) != 64 or file_sha256(weights) != weights_sha256.lower():
        raise ValueError("offline weights SHA256 mismatch")
    with isolated_rng(config["seed"]):
        builder = _model_builder(config["track"])
        if config["track"] == "R":
            model = builder(weights_path=Path(weights), expected_sha256=weights_sha256, head_seed=config["seed"])
        else:
            model = builder(track=config["track"], levir_path=Path(weights), expected_sha256=weights_sha256,
                            head_seed=config["seed"])
    validate_initialized_model(model, config["track"], weights_sha256)
    return model


def make_pilot_optimizer(model, config, phase):
    optimizer_coverage(model)
    pretrained, new = model.parameter_groups()
    if phase == "overfit":
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    if config["track"] == "R":
        return torch.optim.SGD([{"params": pretrained}, {"params": new}], lr=.01, momentum=.9,
                               weight_decay=0., nesterov=False)
    return torch.optim.AdamW([{"params": pretrained, "lr": 1e-4}, {"params": new, "lr": 1e-3}], weight_decay=.05)


def pilot_scheduler(optimizer, track, total_iters):
    if track == "R":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.9, patience=1,
                                                        threshold=1e-4, threshold_mode="rel", cooldown=0,
                                                        min_lr=0., eps=1e-6)
    return make_scheduler(optimizer, total_iters, 1000)


def _image_loss(outputs, labels, track, reduction="mean"):
    if track == "R":
        return F.cross_entropy(outputs["class_logits"].float(), labels.long(), reduction=reduction)
    return F.binary_cross_entropy_with_logits(outputs["attack_logit"].float().reshape(-1),
                                              labels.float().reshape(-1), reduction=reduction)


def train_pilot_epoch(model, loader, optimizer, scheduler, track, device, *, accumulation=None,
                      clip_norm="recipe", max_updates=None):
    model.train()
    accumulation = (2 if track == "R" else 1) if accumulation is None else accumulation
    clip_norm = (None if track == "R" else 5.) if clip_norm == "recipe" else clip_norm
    totals = {"loss": 0., "image_loss": 0., "bce_map": 0., "dice_map": 0.}
    n_images = updates = 0
    iterator = iter(loader)
    while True:
        window = list(itertools.islice(iterator, accumulation))
        if not window or (max_updates is not None and updates >= max_updates):
            break
        window_size = sum(len(batch["label"]) for batch in window)
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            query, labels = batch["query"].to(device), batch["label"].to(device)
            outputs = model(query)
            image_loss = _image_loss(outputs, labels, track)
            parts = {"bce_map": 0., "dice_map": 0.}
            loss = image_loss
            if track == "N":
                if "mask" not in batch or "valid" not in batch:
                    raise ValueError("N training requires mask and valid targets")
                loss, parts = referral_loss(outputs["attack_logit"], outputs["map_logit"], labels,
                                            batch["mask"].to(device), batch["valid"].to(device), 3.)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            count = len(labels)
            (loss * (count / window_size)).backward()
            totals["loss"] += loss.item() * count
            totals["image_loss"] += image_loss.item() * count
            for key in ("bce_map", "dice_map"):
                totals[key] += parts[key] * count
            n_images += count
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError("non-finite training gradient")
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None and track != "R":
            scheduler.step()
        updates += 1
    if not n_images:
        raise ValueError("training loader has no complete batches")
    result = {key: value / n_images for key, value in totals.items()}
    if track != "N":
        result.update(bce_map=None, dice_map=None)
    result.update(n_images=n_images, optimizer_steps=updates)
    return result


def evaluate_pilot(model, loader, track, device):
    if track == "D" and getattr(loader, "batch_size", 1) != 1:
        raise ValueError("D evaluation batch size must be one")
    model.eval()
    before = tensor_sha256(dict(model.named_buffers()))
    predictions, loss_sum = [], 0.
    with torch.inference_mode():
        for batch in loader:
            labels = batch["label"].to(device)
            if track == "D" and len(labels) != 1:
                raise ValueError("D evaluation batch size must be one")
            outputs = model(batch["query"].to(device))
            logits = outputs["attack_logit"].float().reshape(-1)
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite evaluation logits")
            loss_sum += _image_loss(outputs, labels, track, "sum").item()
            probabilities = logits.sigmoid().cpu().tolist()
            raw_logits = logits.cpu().tolist()
            for index, probability in enumerate(probabilities):
                meta = {key: str(batch["meta"][key][index]) for key in ("image_id", "design_id", "attack_family")}
                row = dict(meta, label=int(labels[index].item()), attack_logit=raw_logits[index],
                           attack_probability=probability, bonafide_score=1. - probability)
                if track == "R":
                    row.update(class_logit_0=float(outputs["class_logits"][index, 0]),
                               class_logit_1=float(outputs["class_logits"][index, 1]))
                predictions.append(row)
    if before != tensor_sha256(dict(model.named_buffers())):
        raise ValueError("evaluation changed model buffers")
    if not predictions:
        raise ValueError("evaluation loader has no images")
    if len({row["image_id"] for row in predictions}) != len(predictions):
        raise ValueError("duplicate evaluation image IDs")
    summary = summarize_predictions(predictions)
    summary.update(image_loss=loss_sum / len(predictions), buffers_unchanged=True)
    return summary, predictions


def summarize_predictions(predictions):
    if any(row["label"] == 1 and row["attack_family"] not in ATTACK_SUBGROUPS.values() for row in predictions):
        raise ValueError("unknown SIDTD attack family in predictions")
    scores = [row["bonafide_score"] for row in predictions]
    labels = [row["label"] for row in predictions]
    genuine = [row for row in predictions if row["label"] == 0]
    summary = {"n_images": len(predictions), "n_genuine": labels.count(0), "n_altered": labels.count(1),
               "accuracy": M.deepid_acc(scores, labels), "auc": M.auc(scores, labels),
               "genuine_false_positives": [row["image_id"] for row in genuine if row["bonafide_score"] < .5],
               "altered_misses": [row["image_id"] for row in predictions if row["label"] == 1 and row["bonafide_score"] >= .5],
               "subgroups": {}}
    for family, record_family in ATTACK_SUBGROUPS.items():
        altered = [row for row in predictions if row["label"] == 1 and row["attack_family"] == record_family]
        population = genuine + altered
        family_scores, family_labels = [row["bonafide_score"] for row in population], [row["label"] for row in population]
        summary["subgroups"][family] = {"n_genuine": len(genuine), "n_altered": len(altered),
                                          "auc": M.auc(family_scores, family_labels),
                                          "accuracy": M.deepid_acc(family_scores, family_labels) if altered and genuine else None}
    return summary


def select_phase_rows(rows, phase, role):
    if role not in ("train", "val"):
        raise ValueError("pilot role must be train or val, never test")
    if not rows.loc[rows.label == 1, "attack_family"].isin(ATTACK_SUBGROUPS.values()).all():
        raise ValueError("unknown SIDTD attack family in source membership")
    rows = rows.sort_values("image_id").reset_index(drop=True)
    if phase == "overfit":
        if role != "train":
            raise ValueError("overfit opens training rows only")
        selections = []
        for family, count in (("real", 8), ("text_inpaint", 6), ("text_copy", 2)):
            candidates = rows[rows.label == 0] if family == "real" else rows[(rows.label == 1) & (rows.attack_family == family)]
            if len(candidates) < count:
                raise ValueError(f"overfit requires {count} {family} train images")
            selections.append(candidates.head(count))
        import pandas as pd
        return pd.concat(selections).sort_values("image_id").reset_index(drop=True)
    if phase == "smoke":
        if len(rows) < 32:
            raise ValueError(f"smoke requires 32 {role} images")
        return rows.head(32).copy()
    return rows


def save_pilot_checkpoint(path, model, config, identities, epoch, val_key, phase):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "architecture_id": model.architecture_id, "config": config,
                "config_sha256": json_sha256(config), "identities": identities, "epoch": epoch,
                "selection_key": _selection_record(val_key), "phase": phase, "state_dict": model.state_dict(),
                "initialization_provenance": model.initialization_provenance}, path)


def read_pilot_checkpoint(path, config, identities):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if checkpoint.get("architecture_id") != config.get("architecture_id"):
        raise ValueError("checkpoint architecture mismatch")
    if checkpoint.get("config_sha256") != json_sha256(config) or checkpoint.get("config") != config:
        raise ValueError("checkpoint effective config hash mismatch")
    if checkpoint.get("identities") != identities:
        raise ValueError("checkpoint data/source/weights identities hash mismatch")
    if checkpoint.get("phase") not in ("overfit", "smoke", "train"):
        raise ValueError("checkpoint phase mismatch")
    return checkpoint


def _selection_record(key):
    return [value if math.isfinite(value) else None for value in key]


def compare_replay(expected, actual):
    result = {"passed": False, "atol": ATOL, "rtol": RTOL, "n_images": len(actual)}
    if [row["image_id"] for row in expected] != [row["image_id"] for row in actual]:
        return dict(result, failure="replay image IDs/order mismatch")
    keys = ["attack_logit"]
    if expected and "class_logit_0" in expected[0]:
        keys += ["class_logit_0", "class_logit_1"]
    max_difference = 0.
    for key in keys:
        wanted = np.asarray([float(row[key]) for row in expected])
        observed = np.asarray([float(row[key]) for row in actual])
        if not np.isfinite(wanted).all() or not np.isfinite(observed).all() or not np.allclose(wanted, observed, atol=ATOL, rtol=RTOL):
            return dict(result, failure=f"replay {key} tolerance violation")
        max_difference = max(max_difference, float(np.max(np.abs(wanted - observed), initial=0.)))
    return dict(result, passed=True, max_abs_difference=max_difference)


def check_inference_contract(model, queries, track):
    contract_for(track)
    if len(queries) < 8:
        raise ValueError("inference gate requires eight distinct query images")
    modes = {module: module.training for module in model.modules()}
    before = tensor_sha256(dict(model.named_buffers()))
    model.eval()
    policies = {name: {"kind": type(module).__name__, "training": module.training,
                       "track_running_stats": getattr(module, "track_running_stats", None),
                       "drop_probability": getattr(module, "p", getattr(module, "drop_prob", None))}
                for name, module in model.named_modules()
                if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm,
                                       nn.LayerNorm, nn.GroupNorm, nn.modules.dropout._DropoutNd)) or hasattr(module, "drop_prob")}
    comparisons, max_difference = 0, 0.
    invariant = True
    batch_sizes = [1] if track == "D" else [1, 2, 4, 8]
    try:
        with torch.inference_mode():
            singles = [model(query[None])["attack_logit"].float().reshape(-1)[0].clone() for query in queries[:8]]
            for index in range(8):
                companions = [other for other in range(8) if other != index]
                for batch_size in batch_sizes:
                    positions = sorted({0, batch_size // 2, batch_size - 1})
                    for position in positions:
                        membership = companions[:batch_size - 1]
                        membership.insert(position, index)
                        logit = model(queries[membership])["attack_logit"].float().reshape(-1)[position]
                        difference = float((logit - singles[index]).abs())
                        max_difference = max(max_difference, difference)
                        invariant = invariant and bool(torch.isfinite(logit)) and bool(torch.isclose(logit, singles[index], atol=ATOL, rtol=RTOL))
                        comparisons += 1
    finally:
        for module, mode in modes.items():
            module.training = mode
    after = tensor_sha256(dict(model.named_buffers()))
    result = {"passed": invariant and before == after, "batch_sizes": batch_sizes, "comparisons": comparisons,
            "buffers_unchanged": before == after, "buffer_sha256_before": before, "buffer_sha256_after": after,
            "companion_invariance": invariant if track != "D" else None, "atol": ATOL, "rtol": RTOL,
            "max_abs_difference": max_difference, "normalization_dropout": policies}
    if track == "N":
        result["traversal"] = _check_n_traversals(model, queries)
        result["passed"] = result["passed"] and result["traversal"]["passed"]
    return result


def source_sha256():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src").rglob("*.py")) + [root / "scripts" / "check_anchor_pilots.py"]
    return json_sha256({str(path.relative_to(root)): file_sha256(path) for path in paths if path.is_file()})


def configure_precision(device, track="R"):
    contract_for(track)
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=track == "N")
    return {"dtype": "float32", "amp": False, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": True,
            "deterministic_warn_only": track == "N", "training_bit_reproducible": False,
            "known_nondeterministic_ops": ["grid_sample_cuda_backward"] if track == "N" else [],
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def probe_cuda_grid_sample_policy(device, track):
    result = {"performed": False, "operation": "grid_sample_cuda_backward", "warnings": []}
    if track != "N" or torch.device(device).type != "cuda":
        return result
    enabled, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    def backward():
        inputs = torch.arange(16., device=device).reshape(1, 1, 4, 4).requires_grad_()
        grid = torch.zeros(1, 2, 2, 2, device=device, requires_grad=True)
        F.grid_sample(inputs, grid, align_corners=True).sum().backward()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            backward()
            result["strict_error"] = None
        except RuntimeError as error:
            if "deterministic" not in str(error):
                raise
            result["strict_error"] = str(error)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            backward()
        result.update(performed=True, warn_only_backward_passed=True, warnings=[str(warning.message) for warning in observed])
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return result


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def _loader(dataset, batch_size, workers, seed, *, indices=None, drop_last=False):
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=workers,
                      worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(seed))


def _rows_manifest(rows):
    keys = [key for key in ("image_id", "label", "design_id", "attack_family", "component_id", "path") if key in rows]
    return rows[keys].to_dict(orient="records")


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot export empty predictions/history")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_predictions(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _bn_snapshot(model):
    return {name: int(module.num_batches_tracked.item())
            for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
            and module.num_batches_tracked is not None}


def _training_bn_gate(before, model, expected_calls):
    after = _bn_snapshot(model)
    increments = {name: after[name] - count for name, count in before.items()}
    return {"passed": all(count == expected_calls for count in increments.values()), "increments": increments,
            "expected_forward_calls": expected_calls}


def validate_pretrain_initialization(diagnostic, identities, initialization):
    if (diagnostic.get("schema_version") != SCHEMA_VERSION or diagnostic.get("phase") != "pretrain"
            or diagnostic.get("passed") is not True or diagnostic.get("failures")
            or diagnostic.get("identities") != identities
            or any(diagnostic.get("evidence", {}).get(phase, {}).get("passed") is not True for phase in ("overfit", "smoke"))):
        raise ValueError("completed pretrain diagnostic gate with matching identities is required")
    expected = diagnostic.get("fresh_initialization", {})
    if any(expected.get(key) != initialization[key] for key in ("state_sha256", "buffer_sha256", "pretrained_tensor_sha256")):
        raise ValueError("long-run initialization differs from diagnostic pristine tensors")


def _check_n_traversals(model, queries):
    if not hasattr(model, "backbone") or not hasattr(model, "decode_head"):
        return {"passed": False, "failure": "N traversal instrumentation requires backbone and decode_head"}
    counters = {"query_stem": 0, "shared_projection": 0}
    def stem_hook(module, inputs):
        counters["query_stem"] += 1
    handle = model.backbone.stem.register_forward_pre_hook(stem_hook)
    decoder = model.decode_head
    original = decoder.base_forward
    had_override = "base_forward" in decoder.__dict__
    def projection(*args, **kwargs):
        counters["shared_projection"] += 1
        return original(*args, **kwargs)
    decoder.base_forward = projection
    modes = {module: module.training for module in model.modules()}
    model.eval()
    before = tensor_sha256(model.state_dict())
    try:
        with torch.inference_mode():
            model(queries[:1])
    finally:
        handle.remove()
        if had_override:
            decoder.base_forward = original
        else:
            del decoder.base_forward
        for module, mode in modes.items():
            module.training = mode
    return {"passed": counters == {"query_stem": 1, "shared_projection": 1} and before == tensor_sha256(model.state_dict()),
            "calls": counters, "state_unchanged": before == tensor_sha256(model.state_dict())}


def _inference_gate(model, dataset, track, device):
    queries = torch.stack([dataset[index]["query"] for index in range(8)]).to(device)
    result = check_inference_contract(model, queries, track)
    return result


def run_pilot(config_path, raw_root, legacy_root, run_dir, device, phase, *, weights, weights_sha256,
              records=None, split=None, checkpoint=None, workers=0, pretrain_gate=None):
    from .data.query import AnchorQueryDataset, load_role_rows
    from .pilot_config import resolve_pilot_config

    if phase not in ("overfit", "smoke", "train", "replay"):
        raise ValueError("unknown pilot phase")
    if phase != "replay" and checkpoint is not None:
        raise ValueError("training phases forbid checkpoint initialization")
    config = resolve_pilot_config(Path(config_path))
    validate_weight_request(config["track"], weights_sha256)
    if phase == "train" and config["epochs"] != 100:
        raise ValueError("full pilot training requires 100 epochs without early stopping")
    if config["seed"] != 42 or config["fold"] != 0:
        raise ValueError("only the authorized fold zero seed 42 pilot may execute")
    if workers < 0:
        raise ValueError("worker count must be nonnegative")
    run_dir = Path(run_dir)
    if phase != "replay" and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("pilot run directory must be empty; historical artifacts cannot be overwritten")
    if phase == "replay" and not (run_dir / "train.json").is_file():
        raise ValueError("replay requires an existing versioned pilot run")
    records = Path(records or config["records_csv"])
    split = Path(split or config["split_csv"])
    identities = {"source_sha256": source_sha256(), "config_sha256": json_sha256(config),
                  "records_sha256": file_sha256(records), "split_sha256": file_sha256(split),
                  "weights_sha256": file_sha256(weights)}
    if identities["weights_sha256"] != weights_sha256.lower():
        raise ValueError("offline weights SHA256 mismatch")
    precision = configure_precision(device, config["track"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    track = config["track"]
    if phase == "replay":
        info = _read_json(run_dir / "train.json")
        if info.get("identities") != identities or info.get("config") != config:
            raise ValueError("replay run identities/config hash mismatch")
        saved = read_pilot_checkpoint(checkpoint or run_dir / "checkpoints" / "best.pt", config, identities)
        if saved["phase"] != info["phase"]:
            raise ValueError("replay checkpoint phase mismatch")
        role = "train" if saved["phase"] == "overfit" else "val"
        rows = select_phase_rows(load_role_rows(records, split, role), saved["phase"], role)
        if _rows_manifest(rows) != info["membership"][role]:
            raise ValueError("replay membership manifest mismatch")
        dataset = AnchorQueryDataset(rows, Path(raw_root), legacy_root, track, role == "train", {"enabled": False})
        model = build_initialized_model(config, weights, weights_sha256).to(device)
        if model.initialization_provenance != saved.get("initialization_provenance"):
            raise ValueError("replay initialization provenance mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        summary, predictions = evaluate_pilot(model, _loader(dataset, config["eval_batch_size"], workers,
                                                             config["seed"] + 2001), track, device)
        result = compare_replay(_read_predictions(run_dir / "val_predictions.csv"), predictions)
        result.update(schema_version=SCHEMA_VERSION, phase="replay", identities=identities, pid=os.getpid(),
                      checkpoint_sha256=file_sha256(checkpoint or run_dir / "checkpoints" / "best.pt"),
                      inference=_inference_gate(model, dataset, track, device), summary=summary)
        result["passed"] = result["passed"] and result["inference"]["passed"]
        write_json(run_dir / "replay.json", result)
        return result
    roles = ("train",) if phase == "overfit" else ("train", "val")
    role_rows = {role: select_phase_rows(load_role_rows(records, split, role), phase, role) for role in roles}
    for role, rows in role_rows.items():
        if "role" in rows and not rows.role.eq(role).all():
            raise ValueError("forbidden test or mismatched role in pilot membership")
    if "val" in role_rows and set(role_rows["train"].image_id) & set(role_rows["val"].image_id):
        raise ValueError("train/validation membership overlaps")
    if phase == "train" and {role: len(rows) for role, rows in role_rows.items()} != {"train": 1778, "val": 222}:
        raise ValueError("full pilot requires official 1778 train and 222 validation images")
    train_rows = role_rows["train"]
    train_dataset = AnchorQueryDataset(train_rows, Path(raw_root), legacy_root, track, True,
                                      {"enabled": False} if phase == "overfit" else config["augment"])
    eval_rows = role_rows["train"] if phase == "overfit" else role_rows["val"]
    eval_dataset = AnchorQueryDataset(eval_rows, Path(raw_root), legacy_root, track, phase == "overfit", {"enabled": False})
    val_loader = _loader(eval_dataset, config["eval_batch_size"], workers, config["seed"] + 2001)
    model = build_initialized_model(config, weights, weights_sha256).to(device)
    initial_state = tensor_sha256(model.state_dict())
    initial_buffers = tensor_sha256(dict(model.named_buffers()))
    initialization = validate_initialized_model(model, track, weights_sha256)
    fresh_initialization = {"passed": True, "state_sha256": initial_state, "buffer_sha256": initial_buffers,
                            "pretrained_tensor_sha256": model.initialization_provenance["pretrained_tensor_sha256"],
                            "source": "original_offline_weights_and_seeded_head", "checkpoint_supplied": False}
    if phase == "train":
        if pretrain_gate is None:
            raise ValueError("full training requires --pretrain-gate for pristine initialization comparison")
        diagnostic = _read_json(pretrain_gate)
        validate_pretrain_initialization(diagnostic, identities, fresh_initialization)
        fresh_initialization["pretrain_gate_sha256"] = file_sha256(pretrain_gate)
    optimizer = make_pilot_optimizer(model, config, phase)
    batch_size = 8 if phase == "overfit" else config["batch_size"]
    drop_last = False if phase == "overfit" else config["drop_last"]
    n_batches = len(train_dataset) // batch_size if drop_last else (len(train_dataset) + batch_size - 1) // batch_size
    epochs = 1 if phase == "smoke" else config["epochs"]
    scheduler = None if phase == "overfit" else pilot_scheduler(optimizer, track, epochs * n_batches)
    sampler = torch.Generator().manual_seed(config["seed"] + 1001)
    info = {"schema_version": SCHEMA_VERSION, "architecture_id": model.architecture_id, "track": track,
            "phase": phase, "config": config, "identities": identities, "precision": precision, "device": str(device),
            "phase_recipe": {"batch_size": batch_size, "drop_last": drop_last,
                             "accumulation_steps": 1 if phase == "overfit" else config["accumulation_steps"],
                             "augment": {"enabled": False} if phase == "overfit" else config["augment"],
                             "gradient_clip_norm": None if phase == "overfit" or track == "R" else 5.,
                             "epochs": None if phase == "overfit" else epochs,
                             "max_updates": 500 if phase == "overfit" else None,
                             "evaluation_interval_updates": 50 if phase == "overfit" else None},
            "paths": {"config": str(Path(config_path).resolve()), "records": str(records.resolve()), "split": str(split.resolve()),
                      "weights": str(Path(weights).resolve()), "raw_root": str(Path(raw_root).resolve()),
                      "legacy_root": str(Path(legacy_root).resolve()) if legacy_root else None},
            "membership": {role: _rows_manifest(rows) for role, rows in role_rows.items()},
            "n_images": {role: len(rows) for role, rows in role_rows.items()}, "opened_roles": list(roles), "no_test_open": True,
            "initialization": initialization, "fresh_initialization": fresh_initialization,
            "optimizer_coverage": optimizer_coverage(model),
            "cuda_backward_probe": probe_cuda_grid_sample_policy(device, track) if phase == "overfit" else {"performed": False},
            "optimizer": {"name": "adamw" if phase == "overfit" else config["optimizer"]["name"],
                          "initial_state_empty": len(optimizer.state) == 0,
                          "groups": [{"lr": group["lr"], "weight_decay": group["weight_decay"], "n_tensors": len(group["params"])}
                                     for group in optimizer.param_groups]},
            "scheduler": {"name": "none" if phase == "overfit" else config["scheduler"]["name"],
                          "total_iters": None if phase == "overfit" else epochs * n_batches,
                          "effective_warmup_iters": min(1000, int(.1 * epochs * n_batches)) if track != "R" and phase != "overfit" else 0},
            "rng": {"seed": config["seed"], "sampler_seed": config["seed"] + 1001, "worker_seed": config["seed"] + 2001,
                    "workers": workers, "persistent_workers": False, "head_rng_isolated": True,
                    "sampler_initial_state_sha256": tensor_sha256({"rng": sampler.get_state()})},
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__)},
            "n_active_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "n_total_parameters": sum(parameter.numel() for parameter in model.parameters()), "pid": os.getpid(),
            "omitted_train_ids": [], "bn_training": [], "passed": False}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "train.json", info)
    history = []
    if phase != "overfit":
        info["inference_before"] = _inference_gate(model, eval_dataset, track, device)
        if not info["inference_before"]["passed"]:
            write_json(run_dir / "train.json", info)
            return info
    if phase == "overfit":
        train_loader = _loader(train_dataset, 8, workers, config["seed"] + 2001)
        updates = 0
        while updates < 500:
            before = _bn_snapshot(model)
            stats = train_pilot_epoch(model, train_loader, optimizer, None, track, device,
                                      accumulation=1, clip_norm=None, max_updates=min(2, 500 - updates))
            updates += stats["optimizer_steps"]
            info["bn_training"].append(_training_bn_gate(before, model, stats["optimizer_steps"]))
            if updates % 50 != 0:
                continue
            summary, predictions = evaluate_pilot(model, val_loader, track, device)
            correct = round(summary["accuracy"] * 16)
            history.append({"update": updates, **stats, "eval_image_loss": summary["image_loss"], "n_correct": correct})
            if correct >= 15 and summary["image_loss"] < .15:
                break
        info["overfit"] = dict(summary, n_correct=correct, total_training_loss=stats["loss"], bce_map=stats["bce_map"],
                               dice_map=stats["dice_map"], criterion="n_correct>=15 and mean_image_loss<0.15")
        info.update(updates=updates, passed=correct >= 15 and summary["image_loss"] < .15 and all(item["passed"] for item in info["bn_training"]))
        save_pilot_checkpoint(run_dir / "checkpoints" / "best.pt", model, config, identities, updates, (), phase)
    else:
        best_key, best_epoch = None, 0
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(train_dataset), generator=sampler).tolist()
            kept = len(order) // batch_size * batch_size if drop_last else len(order)
            info["omitted_train_ids"].append({"epoch": epoch, "image_ids": train_rows.iloc[order[kept:]].image_id.tolist()})
            train_loader = _loader(train_dataset, batch_size, workers, config["seed"] + 2001 + epoch,
                                   indices=order, drop_last=drop_last)
            before = _bn_snapshot(model)
            stats = train_pilot_epoch(model, train_loader, optimizer, scheduler, track, device)
            bn_gate = _training_bn_gate(before, model, n_batches)
            info["bn_training"].append(bn_gate)
            if not bn_gate["passed"]:
                raise ValueError("training BN counters violate single-query traversal")
            summary, predictions = evaluate_pilot(model, val_loader, track, device)
            if track == "R":
                scheduler.step(summary["image_loss"])
            key = selection_key("accuracy", summary["accuracy"], summary["auc"], summary["image_loss"], stats["loss"])
            history.append({"epoch": epoch, **stats, "val_image_loss": summary["image_loss"], "val_accuracy": summary["accuracy"],
                            "val_auc": summary["auc"], "lr_pretrained": optimizer.param_groups[0]["lr"],
                            "lr_new": optimizer.param_groups[-1]["lr"]})
            _write_csv(run_dir / "log.csv", history)
            save_pilot_checkpoint(run_dir / "checkpoints" / "last.pt", model, config, identities, epoch, key, phase)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                save_pilot_checkpoint(run_dir / "checkpoints" / "best.pt", model, config, identities, epoch, key, phase)
        selected = read_pilot_checkpoint(run_dir / "checkpoints" / "best.pt", config, identities)
        model.load_state_dict(selected["state_dict"], strict=True)
        summary, predictions = evaluate_pilot(model, val_loader, track, device)
        info["inference_after"] = _inference_gate(model, eval_dataset, track, device)
        info.update(epochs_completed=epochs, selected_epoch=best_epoch, selection_key=_selection_record(best_key),
                    passed=info["inference_after"]["passed"])
    info["selected_checkpoint_sha256"] = file_sha256(run_dir / "checkpoints" / "best.pt")
    _write_csv(run_dir / "log.csv", history)
    _write_csv(run_dir / "val_predictions.csv", predictions)
    write_json(run_dir / "val_summary.json", dict(summary, role="train" if phase == "overfit" else "val",
                                                  selected_epoch=info.get("selected_epoch"),
                                                  localization_metrics=None))
    write_json(run_dir / "train.json", info)
    write_json(run_dir / "gate.json", {"schema_version": SCHEMA_VERSION, "phase": phase, "passed": info["passed"],
                                       "identities": identities, "fresh_initialization": fresh_initialization,
                                       "failures": [] if info["passed"] else ["phase correctness/overfit gate failed"]})
    return info


def fresh_process_replay(*, run_dir, config_path, raw_root, legacy_root, weights, weights_sha256,
                         device, records=None, split=None, workers=0, checkpoint=None):
    command = [sys.executable, "-m", "refer_or_clear.pilots", "--phase", "replay", "--config", str(config_path),
               "--raw-root", str(raw_root), "--run-dir", str(run_dir), "--weights", str(weights),
               "--weights-sha256", weights_sha256, "--device", str(device), "--workers", str(workers)]
    for flag, value in (("--legacy-root", legacy_root), ("--records", records), ("--split", split), ("--checkpoint", checkpoint)):
        if value is not None:
            command.extend([flag, str(value)])
    environment = dict(os.environ)
    source_path = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    if completed.returncode != 0:
        result = {"schema_version": SCHEMA_VERSION, "phase": "replay", "passed": False,
                  "failure": "fresh replay process failed", "returncode": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr}
    else:
        result = _read_json(Path(run_dir) / "replay.json")
        result["fresh_process"] = result.get("pid") != os.getpid()
        result["parent_pid"] = os.getpid()
        result["passed"] = result["passed"] and result["fresh_process"]
    write_json(Path(run_dir) / "replay.json", result)
    return result


def check_pilot_gates(run_dir, phase):
    run_dir = Path(run_dir)
    if phase not in ("pretrain", "selected"):
        raise ValueError("gate phase must be pretrain or selected")
    failures, evidence = [], {}
    phases = ("overfit", "smoke") if phase == "pretrain" else ("train",)
    for name in phases:
        directory = run_dir / name if phase == "pretrain" else run_dir
        try:
            info = _read_json(directory / "train.json")
            if info.get("schema_version") != SCHEMA_VERSION or info.get("phase") != name:
                failures.append(f"{name}: unknown schema/phase")
            contract = contract_for(info["track"])
            if info["architecture_id"] != contract.architecture_id:
                failures.append(f"{name}: architecture mismatch")
            identities, paths = info["identities"], info["paths"]
            if identities.get("config_sha256") != json_sha256(info["config"]) or identities.get("source_sha256") != source_sha256():
                failures.append(f"{name}: effective config/source hash mismatch")
            for key in ("records", "split", "weights"):
                if identities.get(f"{key}_sha256") != file_sha256(paths[key]):
                    failures.append(f"{name}: {key} hash mismatch")
            validate_weight_request(info["track"], identities["weights_sha256"])
            _validate_provenance(info["initialization"]["provenance"], info["track"], identities["weights_sha256"])
            saved = read_pilot_checkpoint(directory / "checkpoints" / "best.pt", info["config"], identities)
            if saved["initialization_provenance"] != info["initialization"]["provenance"]:
                failures.append(f"{name}: initialization provenance checkpoint mismatch")
            if file_sha256(directory / "checkpoints" / "best.pt") != info.get("selected_checkpoint_sha256"):
                failures.append(f"{name}: selected checkpoint hash mismatch")
            for key in ("passed", "no_test_open"):
                if info.get(key) is not True:
                    failures.append(f"{name}: {key} failed")
            for key in ("initialization", "optimizer_coverage", "fresh_initialization"):
                if not info.get(key, {}).get("passed"):
                    failures.append(f"{name}: missing/failed {key}")
            if not info.get("optimizer", {}).get("initial_state_empty"):
                failures.append(f"{name}: optimizer was not fresh")
            if info.get("opened_roles") != (["train"] if name == "overfit" else ["train", "val"]):
                failures.append(f"{name}: forbidden role access")
            precision = info.get("precision", {})
            if precision.get("dtype") != "float32" or any(precision.get(key) is not False for key in ("amp", "matmul_tf32", "cudnn_tf32")):
                failures.append(f"{name}: strict FP32 missing")
            if precision.get("deterministic_warn_only") != (info["track"] == "N"):
                failures.append(f"{name}: deterministic-kernel policy missing/mismatched")
            if name == "overfit" and info["track"] == "N" and info.get("device", "").startswith("cuda"):
                if not info.get("cuda_backward_probe", {}).get("warn_only_backward_passed"):
                    failures.append("overfit: CUDA grid_sample backward probe missing/failed")
            if not info.get("bn_training") or not all(item.get("passed") for item in info["bn_training"]):
                failures.append(f"{name}: BN training audit missing/failed")
            if name == "overfit":
                probe = info.get("overfit", {})
                if probe.get("n_correct", 0) < 15 or probe.get("image_loss", 1.) >= .15 or not 0 < info.get("updates", 0) <= 500:
                    failures.append("overfit: bounded image classification criterion failed")
                if info.get("n_images") != {"train": 16}:
                    failures.append("overfit: incorrect fixed image count")
            else:
                expected_epochs, expected_counts = (1, {"train": 32, "val": 32}) if name == "smoke" else (100, {"train": 1778, "val": 222})
                if info.get("epochs_completed") != expected_epochs or info.get("n_images") != expected_counts:
                    failures.append(f"{name}: epoch/image count mismatch")
                if saved["epoch"] != info.get("selected_epoch") or saved["selection_key"] != info.get("selection_key"):
                    failures.append(f"{name}: selected epoch/key mismatch")
                predictions = _read_predictions(directory / "val_predictions.csv")
                if [row["image_id"] for row in predictions] != [row["image_id"] for row in info["membership"]["val"]]:
                    failures.append(f"{name}: prediction membership mismatch")
                for key in ("inference_before", "inference_after"):
                    if not info.get(key, {}).get("passed"):
                        failures.append(f"{name}: missing/failed {key}")
                replay = _read_json(directory / "replay.json")
                if not replay.get("passed") or not replay.get("fresh_process") or not replay.get("inference", {}).get("passed"):
                    failures.append(f"{name}: fresh-process replay/invariance failed")
                if replay.get("identities") != info.get("identities") or replay.get("checkpoint_sha256") != info.get("selected_checkpoint_sha256"):
                    failures.append(f"{name}: replay identities/checkpoint mismatch")
            evidence[name] = info
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{name}: incomplete evidence ({error})")
    if len(evidence) == len(phases):
        first = evidence[phases[0]]
        for info in evidence.values():
            if info["identities"] != first["identities"] or info["fresh_initialization"] != first["fresh_initialization"]:
                failures.append("diagnostic identity/pristine initialization mismatch")
    selected = evidence.get("smoke", evidence.get("train", evidence.get("overfit", {})))
    return {"schema_version": SCHEMA_VERSION, "phase": phase, "passed": not failures, "failures": failures,
            "identities": selected.get("identities", {}), "fresh_initialization": selected.get("fresh_initialization", {}),
            "tracks": {key: info.get("track") for key, info in evidence.items()},
            "evidence": {key: {"passed": info.get("passed"), "selected_checkpoint_sha256": info.get("selected_checkpoint_sha256")}
                         for key, info in evidence.items()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("overfit", "smoke", "train", "replay"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pretrain-gate", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.phase == "train" and args.checkpoint is not None:
        raise ValueError("--phase train forbids checkpoint initialization")
    options = vars(args)
    options["config_path"] = options.pop("config")
    result = run_pilot(**options)
    if not result.get("passed"):
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
