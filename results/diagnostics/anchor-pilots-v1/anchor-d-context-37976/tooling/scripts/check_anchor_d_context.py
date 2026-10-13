"""Inference-only D37963 normalization-context assay, not a scientific gate.

Run this sidecar outside the immutable 095 source tree. All paths are explicit
relocation bindings; hashes, scientific helpers, and ordered training membership
must match the original diagnostic. Only the 16 training-probe canvases and their
existing zones are read. No optimizer, selection, augmentation, or target loading
is performed. Outputs are context.json and paired_predictions.csv in a new run
directory. Larger batches use the diagnostic route, never production forward.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

FROZEN_SOURCE_SHA256 = "2f8a833e5cb3450fdccd8c58dba8f07f58af330747a69cca7363d1a21528da9a"
FROZEN_CHECKPOINT_SHA256 = "c78d0b4a06b5dbde434d77f34ffa14ea437dbde4e1d3eafbd2d8aadfff72c47f"
BATCH_SIZES = (1, 4, 8, 16)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "parent-manifest", "train-json", "config", "records", "split", "weights",
                 "checkpoint", "predictions", "raw-root", "legacy-root", "run-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("parent-manifest-sha256", "train-json-sha256", "predictions-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=0, type=int)
    return parser


def parse_args(argv=None):
    return argument_parser().parse_args(argv)


def load_runtime(source_dir):
    root = Path(source_dir).resolve(strict=True) / "src"
    for name, module in list(sys.modules.items()):
        if name == "refer_or_clear" or name.startswith("refer_or_clear."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(root):
                raise ValueError("fresh process required, non-frozen refer_or_clear module already imported")
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    runtime = importlib.import_module("refer_or_clear.pilots")
    if not Path(runtime.__file__).resolve().is_relative_to(root):
        raise ValueError("runtime did not load from declared frozen source")
    return runtime


def _read_json(path):
    return json.loads(Path(path).read_text())


def _checked_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in str(relative):
        raise ValueError("unsafe relative input path")
    root = Path(root).resolve(strict=True)
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root)):
        raise ValueError("symlink input path")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_entries(entries, root, pilots):
    seen = set()
    for entry in entries:
        if entry["path"] in seen:
            raise ValueError("duplicate inventory entry")
        seen.add(entry["path"])
        path = _checked_file(root, entry["path"])
        if path.stat().st_size != entry["size"] or pilots.file_sha256(path) != entry["sha256"]:
            raise ValueError(f"inventory file hash/size mismatch: {entry['path']}")


def validate_bindings(args, pilots, config, rows):
    for name in ("parent_manifest", "train_json", "predictions"):
        if pilots.file_sha256(getattr(args, name)) != getattr(args, f"{name}_sha256"):
            raise ValueError(f"{name} authority hash mismatch")
    parent, info = _read_json(args.parent_manifest), _read_json(args.train_json)
    if (parent.get("schema_version") != "anchor_job_manifest_v1" or parent.get("track") != "D"
            or parent.get("phase") != "diagnostic" or parent.get("allocation", {}).get("job_id") != "37963"
            or parent.get("data_scope") != "train_val_only"):
        raise ValueError("original D37963 diagnostic parent manifest required")
    if (info.get("schema_version") != "anchor_pilot_v1" or info.get("track") != "D"
            or info.get("phase") != "overfit" or info.get("opened_roles") != ["train"]
            or info.get("no_test_open") is not True or info.get("n_images") != {"train": 16}):
        raise ValueError("training-only D overfit ledger required")
    if (config != parent.get("effective_config") or config != info.get("config")
            or config.get("track") != "D" or config.get("seed") != 42 or config.get("fold") != 0
            or config.get("eval_batch_size") != 1 or config.get("bn_policy") != "batch_stats"
            or info.get("architecture_id") != config.get("architecture_id")):
        raise ValueError("effective D config binding mismatch")
    recipe = info.get("phase_recipe", {})
    if any(recipe.get(key) != value for key, value in
           {"batch_size": 8, "drop_last": False, "augment": {"enabled": False}, "accumulation_steps": 1}.items()):
        raise ValueError("original ordered batch-8 overfit recipe required")
    source_files = parent["source_files"]
    provenance = parent["provenance"]
    if pilots.json_sha256(source_files) != provenance["source_sha256"]:
        raise ValueError("parent source inventory authority mismatch")
    _verify_entries(source_files, args.source_dir, pilots)
    identities = {"source_sha256": pilots.source_sha256(), "config_sha256": pilots.json_sha256(config),
                  **{f"{name}_sha256": pilots.file_sha256(getattr(args, name)) for name in ("records", "split", "weights")}}
    if identities["source_sha256"] != FROZEN_SOURCE_SHA256:
        raise ValueError("scientific source must be frozen 095")
    if identities != info.get("identities") or identities != parent.get("runtime_identities"):
        raise ValueError("runtime source/config/records/split/weights identities mismatch")
    if any(provenance.get(f"{name}_sha256") != identities[f"{name}_sha256"] for name in ("records", "split", "weights")):
        raise ValueError("parent metadata/weights provenance mismatch")
    if pilots.file_sha256(args.config) != provenance["config_sha256"]:
        raise ValueError("config file hash mismatch")
    pilots.validate_weight_request("D", identities["weights_sha256"])
    checkpoint_hash = pilots.file_sha256(args.checkpoint)
    if checkpoint_hash != info.get("selected_checkpoint_sha256") or checkpoint_hash != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("original D37963 overfit checkpoint hash mismatch")
    if len(rows) != 16 or not rows.role.eq("train").all() or rows.image_id.duplicated().any():
        raise ValueError("probe must contain exactly 16 distinct training images")
    membership = pilots._rows_manifest(rows)
    if info.get("membership") != {"train": membership}:
        raise ValueError("canonical ordered probe membership mismatch")
    if pilots._rows_manifest(pilots.select_phase_rows(rows, "overfit", "train")) != membership:
        raise ValueError("probe family counts or ordering mismatch")
    parent_roles = parent.get("membership", {})
    identifiers = rows.image_id.tolist()
    if (not set(identifiers).issubset(parent_roles.get("train", []))
            or set(identifiers).intersection(parent_roles.get("val", []))):
        raise ValueError("probe is not exclusively in parent training membership")
    needed = set()
    for row in membership:
        identifier = row["image_id"]
        if Path(identifier).name != identifier or "\\" in identifier:
            raise ValueError("unsafe probe image_id")
        needed.add(("legacy", f"canvases/{identifier}.png"))
        needed.add(("legacy", f"zones/{row['design_id'].replace('/', '_')}.png"))
    data_files = parent["data_files"]
    if pilots.json_sha256(data_files) != provenance["data_sha256"]:
        raise ValueError("parent data inventory authority mismatch")
    indexed = {(entry["kind"], entry["path"]): entry for entry in data_files}
    if len(indexed) != len(data_files) or not needed.issubset(indexed):
        raise ValueError("duplicate or missing probe data inventory")
    verified = [indexed[key] for key in sorted(needed)]
    _verify_entries(verified, args.legacy_root, pilots)
    return {"identities": identities, "membership": membership, "verified_data_files": verified,
            "checkpoint_sha256": checkpoint_hash, "parent_manifest_sha256": args.parent_manifest_sha256,
            "train_json_sha256": args.train_json_sha256, "predictions_sha256": args.predictions_sha256,
            "raw_pixels_read": False, "opened_roles": ["train"], "no_test_open": True}


def _fingerprints(model, pilots):
    return {"parameters_sha256": pilots.tensor_sha256(dict(model.named_parameters())),
            "buffers_sha256": pilots.tensor_sha256(dict(model.named_buffers())),
            "state_sha256": pilots.tensor_sha256(model.state_dict())}


def evaluate_context(model, queries, labels, image_ids, *, pilots, encode_query, expected_predictions=None):
    import torch
    from torch import nn

    if (model.track != "D" or queries.ndim != 4 or queries.shape[:2] != (16, 4)
            or queries.dtype != torch.float32 or len(labels) != 16 or len(image_ids) != 16
            or len(set(image_ids)) != 16 or not torch.isfinite(queries).all()
            or not torch.isin(labels, labels.new_tensor([0, 1])).all()):
        raise ValueError("D context requires 16 distinct FP32 four-channel training probes and binary labels")
    model.eval()
    policies = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.track_running_stats or any(getattr(module, key) is not None for key in
                                                 ("running_mean", "running_var", "num_batches_tracked")):
                raise ValueError("D context requires stateless batch normalization")
            policies.append({"name": name, "kind": type(module).__name__, "training": module.training,
                             "track_running_stats": module.track_running_stats})
    if not policies:
        raise ValueError("D context requires batch normalization")
    before = _fingerprints(model, pilots)
    predictions, summaries = [], {}

    def route(batch):
        if any(module.training for module in model.modules()):
            raise ValueError("diagnostic route requires every module in evaluation mode")
        model._check_bn_policy()
        features = encode_query(model.backbone, batch)
        logits = model.classifier(features[-1].mean(dim=(2, 3))).squeeze(1)
        if logits.dtype != torch.float32 or logits.shape != (len(batch),) or not torch.isfinite(logits).all():
            raise ValueError("diagnostic route produced invalid FP32 logits")
        return logits

    try:
        with torch.inference_mode():
            singleton = []
            public_predictions = []
            max_difference = 0.
            for index in range(16):
                batch = queries[index:index + 1]
                public = model(batch)["attack_logit"].float().reshape(-1)
                diagnostic = route(batch)
                if not torch.isfinite(public).all() or not torch.allclose(public, diagnostic, atol=pilots.ATOL, rtol=pilots.RTOL):
                    raise ValueError("diagnostic route singleton equivalence failed")
                max_difference = max(max_difference, float((public - diagnostic).abs().max()))
                singleton.append(float(diagnostic[0]))
                public_predictions.append({"image_id": image_ids[index], "attack_logit": float(public[0])})
            if expected_predictions is not None:
                if [int(row["label"]) for row in expected_predictions] != labels.cpu().tolist():
                    raise ValueError("saved singleton prediction labels mismatch")
                replay = pilots.compare_replay(expected_predictions, public_predictions)
                if not replay["passed"]:
                    raise ValueError(f"saved singleton replay failed: {replay.get('failure')}")
            else:
                replay = None
            try:
                model(queries[:4])
            except ValueError as error:
                if "batch size one" not in str(error):
                    raise ValueError("public D guard raised an unexpected error") from error
            else:
                raise ValueError("public D singleton evaluation guard is missing")
            for size in BATCH_SIZES:
                records = []
                for start in range(0, 16, size):
                    logits = queries.new_tensor(singleton[start:start + size]) if size == 1 else route(queries[start:start + size])
                    losses = pilots._image_loss({"attack_logit": logits}, labels[start:start + size], "D", "none")
                    for position in range(size):
                        index = start + position
                        logit = float(logits[position])
                        probability = float(logits[position].sigmoid())
                        record = {"image_id": image_ids[index], "label": int(labels[index]), "batch_size": size,
                                  "group_index": start // size, "position": position,
                                  "group_members": image_ids[start:start + size], "attack_logit": logit,
                                  "attack_probability": probability, "image_ce": float(losses[position]),
                                  "correct": int(int(probability > .5) == int(labels[index])),
                                  "singleton_logit": singleton[index], "delta_from_singleton": logit - singleton[index]}
                        records.append(record)
                summaries[str(size)] = {"n_images": 16, "n_correct": sum(row["correct"] for row in records),
                                        "image_ce": sum(row["image_ce"] for row in records) / 16,
                                        "max_abs_delta_from_singleton": max(abs(row["delta_from_singleton"]) for row in records)}
                predictions.extend(records)
    finally:
        after = _fingerprints(model, pilots)
        if before != after:
            raise ValueError("immutable model parameter/buffer fingerprints changed")
    return {"diagnostic_only": True, "diagnostic_complete": True, "scientific_gate_passed": None,
            "batch_sizes": list(BATCH_SIZES), "mode": "eval", "autograd": "inference_mode", "normalization": policies,
            "singleton_equivalence": {"passed": True, "atol": pilots.ATOL, "rtol": pilots.RTOL,
                                      "max_abs_difference": max_difference}, "saved_singleton_replay": replay,
            "public_singleton_guard_preserved": True, "fingerprints_before": before, "fingerprints_after": after,
            "predictions": predictions, "summaries": summaries}


def run(args, pilots):
    import torch
    from refer_or_clear.data.query import AnchorQueryDataset, load_role_rows
    from refer_or_clear.models.anchor_changer import encode_query
    from refer_or_clear.pilot_config import resolve_pilot_config

    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.run_dir.exists():
        raise ValueError("diagnostic run directory must be new")
    for root in (args.source_dir, args.raw_root, args.legacy_root, args.train_json.parent, args.parent_manifest.parent):
        if args.run_dir.resolve().is_relative_to(root.resolve()):
            raise ValueError("output directory must be isolated from frozen inputs")
    config = resolve_pilot_config(args.config)
    rows = pilots.select_phase_rows(load_role_rows(args.records, args.split, "train"), "overfit", "train")
    bindings = validate_bindings(args, pilots, config, rows)
    info = _read_json(args.train_json)
    saved = pilots.read_pilot_checkpoint(args.checkpoint, config, bindings["identities"])
    if saved["phase"] != "overfit" or saved.get("epoch") != info["updates"] or saved.get("selection_key") != []:
        raise ValueError("checkpoint must be the final overfit endpoint without model selection")
    if saved.get("initialization_provenance") != info["initialization"]["provenance"]:
        raise ValueError("checkpoint/ledger initialization provenance mismatch")
    precision = pilots.configure_precision(args.device, "D")
    model = pilots.build_initialized_model(config, args.weights, bindings["identities"]["weights_sha256"]).to(args.device)
    original = dict(saved["initialization_provenance"])
    relocated = dict(model.initialization_provenance)
    original.pop("checkpoint_path", None)
    relocated.pop("checkpoint_path", None)
    if original != relocated:
        raise ValueError("relocated official initialization provenance mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    dataset = AnchorQueryDataset(rows, args.raw_root, args.legacy_root, "D", True, {"enabled": False})
    loader = pilots._loader(dataset, 1, args.workers, config["seed"] + 2001)
    queries, labels, identifiers = [], [], []
    for batch in loader:
        queries.append(batch["query"])
        labels.append(batch["label"])
        identifiers.extend(batch["meta"]["image_id"])
    if identifiers != rows.image_id.tolist():
        raise ValueError("loader changed canonical probe ordering")
    result = evaluate_context(model, torch.cat(queries).to(args.device), torch.cat(labels).to(args.device), identifiers,
                              pilots=pilots, encode_query=encode_query,
                              expected_predictions=pilots._read_predictions(args.predictions))
    singleton = result["summaries"]["1"]
    if (singleton["n_correct"] != info["overfit"]["n_correct"]
            or abs(singleton["image_ce"] - info["overfit"]["image_loss"]) > pilots.ATOL + pilots.RTOL * abs(info["overfit"]["image_loss"])):
        raise ValueError("singleton summary does not reproduce original overfit ledger")
    if validate_bindings(args, pilots, config, rows) != bindings:
        raise ValueError("immutable input bindings changed during diagnostic")
    result.update(schema_version="anchor_d_context_v1", bindings=bindings, precision=precision,
                  source_dir=str(args.source_dir.resolve()), checkpoint_epoch=saved["epoch"],
                  initialization_checkpoint_path_original=saved["initialization_provenance"]["checkpoint_path"],
                  initialization_checkpoint_path_relocated=model.initialization_provenance["checkpoint_path"],
                  diagnostic_script_sha256=pilots.file_sha256(__file__))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    csv_rows = [{**row, "group_members": json.dumps(row["group_members"], separators=(",", ":"))}
                for row in result["predictions"]]
    pilots._write_csv(args.run_dir / "paired_predictions.csv", csv_rows)
    pilots.write_json(args.run_dir / "context.json", result)
    return result


def main(argv=None):
    args = parse_args(argv)
    return run(args, load_runtime(args.source_dir))


if __name__ == "__main__":
    main()
