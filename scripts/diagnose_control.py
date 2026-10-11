#!/usr/bin/env python3
"""Checkpoint-only BN diagnostics on training and validation data, never the test split.

Batch-statistic probes are batch-dependent diagnostics, not deployable benchmark results.
Recalibration uses only native, unaugmented training images and changes buffers, not weights.
Every intervention is isolated and restored; source checkpoints are copied and hashed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@contextmanager
def bn_probe(model, mode):
    """Enable only selected BN batch statistics, without updating any running buffers."""
    if mode not in ("eval", "backbone_batch", "all_batch"):
        raise ValueError(mode)
    flags = [(module, module.training) for module in model.modules()]
    batch_norms = [(name, module, module.track_running_stats)
                   for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        model.eval()
        for name, module, _ in batch_norms:
            enabled = mode == "all_batch" or (mode == "backbone_batch" and name.startswith("backbone."))
            if enabled:
                module.training = True
                module.track_running_stats = False
        with torch.no_grad():
            yield
    finally:
        for _, module, tracking in batch_norms:
            module.track_running_stats = tracking
        for module, training in flags:
            module.training = training


@torch.no_grad()
def recalibrate(model, loader, device, role):
    """Re-estimate shared BN buffers on training inputs only; no gradients or parameter updates."""
    if role != "train":
        raise ValueError("BN calibration must use training data only")
    model.eval()
    batch_norms = [module for module in model.modules()
                   if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    settings = [(module, module.momentum, module.track_running_stats) for module in batch_norms]
    try:
        for module in batch_norms:
            module.reset_running_stats()
            module.momentum = None
            module.track_running_stats = True
            module.training = True
        for batch in loader:
            model(batch["x"].to(device))
    finally:
        for module, momentum, tracking in settings:
            module.momentum = momentum
            module.track_running_stats = tracking
        model.eval()


def summarize(labels, logits):
    """Keep raw-logit ranking alongside the existing float32 bona fide-score convention."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite diagnostic logits")
    scores = (1.0 - torch.sigmoid(torch.tensor(logits, dtype=torch.float32))).numpy()
    both = len(np.unique(labels)) == 2
    result = dict(n=len(labels), auc_logit=float(roc_auc_score(labels, logits)) if both else None,
                  auc_pipeline_score=float(roc_auc_score(labels, -scores)) if both else None,
                  accuracy=float(np.mean((scores < 0.5) == labels)),
                  predicted_forged=int(np.sum(scores < 0.5)),
                  bce_image=float(functional.binary_cross_entropy_with_logits(
                      torch.tensor(logits), torch.tensor(labels, dtype=torch.float64))),
                  unique_scores=int(len(np.unique(scores))))
    for label, name in ((0, "genuine"), (1, "forged")):
        values = logits[labels == label]
        result[name] = dict(n=len(values), logit_mean=float(values.mean()) if len(values) else None,
                            logit_quantiles=np.quantile(values, [0, .05, .5, .95, 1]).tolist() if len(values) else [],
                            predicted_forged=int(np.sum(scores[labels == label] < .5)))
    return result


def score(model, loader, device, mode):
    rows = []
    with bn_probe(model, mode):
        for batch_index, batch in enumerate(loader):
            logits = model(batch["x"].to(device))["logit"].float().reshape(-1).cpu()
            scores = 1.0 - torch.sigmoid(logits)
            for index, logit in enumerate(logits.tolist()):
                rows.append(dict(image_id=batch["meta"]["image_id"][index],
                                 design_id=batch["meta"]["design_id"][index],
                                 label=int(batch["label"][index]), batch=batch_index,
                                 logit_fake=logit, score_bonafide=float(scores[index])))
    return summarize([row["label"] for row in rows], [row["logit_fake"] for row in rows]), rows


@torch.no_grad()
def first_bn_trace(model, batch, device):
    """Observe the two calls to the first shared encoder BN on one fixed training batch."""
    name, module = next((name, module) for name, module in model.named_modules()
                        if name.startswith("backbone.") and isinstance(module, nn.BatchNorm2d))
    calls = []

    def capture(layer, inputs):
        features = inputs[0].detach().float()
        mean = features.mean((0, 2, 3))
        variance = features.var((0, 2, 3), unbiased=False)
        calls.append(dict(call=len(calls), batch_mean=mean.cpu().tolist(),
                          batch_variance=variance.cpu().tolist(),
                          mean_abs_standardized_shift=float(
                              ((mean - layer.running_mean) / torch.sqrt(layer.running_var + layer.eps)).abs().mean())))

    handle = module.register_forward_pre_hook(capture)
    try:
        with bn_probe(model, "eval"):
            model(batch["x"].to(device))
    finally:
        handle.remove()
    return dict(module=name, running_mean=module.running_mean.cpu().tolist(),
                running_variance=module.running_var.cpu().tolist(), calls=calls)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_matches(model, original):
    return all(torch.equal(value.detach().cpu(), original[name]) for name, value in model.state_dict().items())


def main(argv=None):
    from refer_or_clear import evaluate as evaluation
    from refer_or_clear.logging_utils import dump_json, git_hash
    from refer_or_clear.models import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--calibration-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", nargs="+", choices=("eval", "backbone_batch", "all_batch", "train_recalibrated"),
                        default=("eval", "backbone_batch", "all_batch", "train_recalibrated"))
    parser.add_argument("--roles", nargs="+", choices=("train", "val"), default=("train", "val"))
    args = parser.parse_args(argv)
    if min(args.train_limit, args.calibration_limit) < 2 or args.batch_size < 1:
        parser.error("limits must be at least two and batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(args=vars(args), hostname=platform.node(), torch=torch.__version__, git_hash=git_hash(ROOT),
                    script_sha256=sha256(__file__), test_split_used=False, precision="float32",
                    warning="Batch-statistic probes are batch-dependent diagnostics, not benchmark results.")
    dump_json(out / "metadata.json", metadata)
    report = ["# Control checkpoint diagnostic", "", metadata["warning"], "",
              "No test images or test labels are used. No optimizer or gradient updates are run.", "",
              "| Checkpoint | Epoch | Mode | Split | n | AUC logits | AUC pipeline | Accuracy | Image BCE |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for checkpoint_name in args.checkpoints:
        source = Path(checkpoint_name)
        tag = source.parent.name
        target = out / tag / source.stem
        target.mkdir(parents=True)
        source_hash = sha256(source)
        snapshot = target / "checkpoint.pt"
        shutil.copy2(source, snapshot)
        if sha256(snapshot) != source_hash or sha256(source) != source_hash:
            raise RuntimeError("checkpoint changed while snapshotting; wait for training to finish")
        checkpoint = torch.load(snapshot, map_location="cpu")
        resolved = checkpoint["config"]
        if resolved["protocol"] != "A" or resolved["fold"] != 0:
            raise ValueError("this diagnostic is scoped to Protocol A fold 0")
        info_path = Path(args.run_root) / tag / "train.json"
        info = json.loads(info_path.read_text())
        config = info["raw_config"]
        dump_json(target / "train_metadata.json", info)
        split, records = evaluation.load_split(config, args.data_root, fold=resolved["fold"])
        pairs = {role: evaluation.role_pairs(split, records, args.data_root, role, resolved["reference"], config,
                                             resolved["seed"], wrong_design_fraction=0.0)
                 for role in ("train", "val")}
        train_ids = {pair["image_id"] for pair in pairs["train"]}
        val_ids = {pair["image_id"] for pair in pairs["val"]}
        if train_ids & val_ids:
            raise ValueError("training and validation image IDs overlap")
        calibration = evaluation.subsample(pairs["train"], args.calibration_limit, 42)
        pairs["train"] = evaluation.subsample(pairs["train"], args.train_limit, 42)
        for selected in [*pairs.values(), calibration]:
            np.random.RandomState(42).shuffle(selected)
        loaders = {role: evaluation.eval_loader(selected, args.batch_size, args.workers)
                   for role, selected in pairs.items()}
        calibration_loader = evaluation.eval_loader(calibration, args.batch_size, args.workers)
        dump_json(target / "inputs.json", dict(
            scored={role: [pair["image_id"] for pair in selected] for role, selected in pairs.items()},
            calibration_train_ids=[pair["image_id"] for pair in calibration],
            split_sha256=sha256(Path(args.data_root) / "splits" /
                                Path(config["data"]["split"]).name.format(fold=resolved["fold"]))))
        model = build_model(arch=resolved["arch"], fusion=resolved["fusion"], init="scratch", in_channels=4).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        results = dict(source_checkpoint=str(source), source_sha256=source_hash, epoch=checkpoint["epoch"],
                       config=resolved, modes={}, first_bn=first_bn_trace(model, next(iter(loaders["train"])), device))
        expected_ids = {}
        for mode in args.modes:
            model.load_state_dict(original)
            if mode == "train_recalibrated":
                recalibrate(model, calibration_loader, device, role="train")
                if any(not torch.equal(parameter.detach().cpu(), original[name])
                       for name, parameter in model.named_parameters()):
                    raise AssertionError("calibration changed model parameters")
            results["modes"][mode] = {}
            for role in args.roles:
                loader = loaders[role]
                block, rows = score(model, loader, device, "eval" if mode == "train_recalibrated" else mode)
                identifiers = [(row["image_id"], row["label"]) for row in rows]
                if role in expected_ids and identifiers != expected_ids[role]:
                    raise AssertionError("probe sample order or labels changed")
                expected_ids[role] = identifiers
                results["modes"][mode][role] = block
                with open(target / f"{mode}_{role}.csv", "w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                line = (f"| {tag}/{source.stem} | {checkpoint['epoch']} | {mode} | {role} | {block['n']} | "
                        f"{block['auc_logit']:.4f} | {block['auc_pipeline_score']:.4f} | "
                        f"{block['accuracy']:.4f} | {block['bce_image']:.4f} |")
                report.append(line)
                print(line, flush=True)
            if mode != "train_recalibrated" and not state_matches(model, original):
                raise AssertionError("a batch-statistic probe changed the model state")
            dump_json(target / "results.json", results)
            (out / "summary.md").write_text("\n".join(report) + "\n")
        model.load_state_dict(original)
        if not state_matches(model, original) or sha256(source) != source_hash:
            raise AssertionError("original state or source checkpoint changed")
        results["state_restored"] = True
        results["source_checkpoint_unchanged"] = True
        dump_json(target / "results.json", results)
        del model, checkpoint, original, loaders, calibration_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metadata["complete"] = True
    dump_json(out / "metadata.json", metadata)
    print(f"Diagnostic complete: {out}", flush=True)


if __name__ == "__main__":
    main()
