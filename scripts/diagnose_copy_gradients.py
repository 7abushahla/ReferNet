#!/usr/bin/env python3
"""Training-only, fixed-checkpoint gradient geometry in single-image evaluation mode.

This local sensitivity diagnostic does not reproduce minibatch training dynamics.
Negative gradient alignment alone does not establish harmful multitask training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
from statistics import median
import sys

import torch
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def gradient_geometry(image_gradients, localization_gradients):
    entries = [(image, localization) for image, localization in
               zip(image_gradients, localization_gradients) if image is not None or localization is not None]
    if not entries:
        raise ValueError("no gradients")
    device = next(value.device for pair in entries for value in pair if value is not None)
    zero = torch.zeros((), device=device)
    image_square = torch.stack([image.square().sum() if image is not None else zero
                                for image, _ in entries]).sum()
    localization_square = torch.stack([localization.square().sum() if localization is not None else zero
                                       for _, localization in entries]).sum()
    dot = torch.stack([(image * localization).sum() if image is not None and localization is not None else zero
                       for image, localization in entries]).sum()
    image_square, localization_square, dot = torch.stack((image_square, localization_square, dot)).tolist()
    if not all(math.isfinite(value) for value in (image_square, localization_square, dot)):
        raise ValueError("nonfinite gradients")
    image_norm, localization_norm = math.sqrt(image_square), math.sqrt(localization_square)
    return dict(image_norm=image_norm, localization_norm=localization_norm, dot=dot,
                cosine=max(-1., min(1., dot / (image_norm * localization_norm)))
                if image_norm and localization_norm else None,
                localization_to_image_norm=localization_norm / image_norm if image_norm else None)


def select_training_pairs(pairs, role, seed, limit_per_family=None):
    if role != "train":
        raise ValueError("gradient diagnostic requires training data")
    if any(pair["wrong_design"] for pair in pairs):
        raise ValueError("wrong-design pairs are not eligible")
    rng = random.Random(seed)
    copies = sorted((pair for pair in pairs if pair["attack_family"] == "text_copy"),
                    key=lambda pair: pair["image_id"])
    if limit_per_family is not None:
        rng.shuffle(copies)
        copies = copies[:limit_per_family]
    if not copies:
        raise ValueError("no copy-and-replace training examples")
    selected = list(copies)
    for design, count in sorted(Counter(pair["design_id"] for pair in copies).items()):
        for family in ("text_inpaint", ""):
            candidates = sorted((pair for pair in pairs if pair["design_id"] == design
                                 and pair["attack_family"] == family), key=lambda pair: pair["image_id"])
            if len(candidates) < count:
                raise ValueError(f"insufficient {family} examples for {design}")
            selected.extend(rng.sample(candidates, count))
    return sorted(selected, key=lambda pair: pair["image_id"])


def summarize_rows(rows):
    result = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        groups = {}
        for scope in ("backbone", "decode_head", "shared"):
            cosines = [row[f"{scope}_cosine"] for row in selected if row[f"{scope}_cosine"] is not None]
            ratios = [row[f"{scope}_localization_to_image_norm"] for row in selected
                      if row[f"{scope}_localization_to_image_norm"] is not None]
            groups[scope] = dict(median_cosine=median(cosines) if cosines else None,
                                 negative_cosine_fraction=sum(value < 0 for value in cosines) / len(cosines)
                                 if cosines else None,
                                 median_norm_ratio=median(ratios) if ratios else None)
        result[family] = dict(count=len(selected), gradients=groups,
                             errors_at_zero_logit=sum((row["logit_fake"] > 0) != bool(row["label"])
                                                      for row in selected),
                             median_image_loss=median(row["image_loss"] for row in selected),
                             median_weighted_localization_loss=median(row["weighted_localization_loss"]
                                                                      for row in selected))
    return result


def main():
    from refer_or_clear import evaluate as evaluation
    from refer_or_clear.data.dataset import PairDataset
    from refer_or_clear.losses import referral_loss
    from refer_or_clear.train import seed_everything
    from diagnose_control import sha256
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="A-cex-single_noex-median-levir-f0-s42-bnbatch-e100")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-per-family", type=int)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "run on an allocated GPU node"
    seed_everything(42)
    torch.set_num_threads(4)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((Path("results/runs") / args.run / "train.json").read_text())
    config = metadata["raw_config"]
    split, records = evaluation.load_split(config, args.data_root, fold=0)
    pairs = evaluation.role_pairs(split, records, args.data_root, "train", "median", config, 42,
                                   wrong_design_fraction=0.)
    assert len(pairs) == 1778 and len({pair["image_id"] for pair in pairs}) == 1778
    selected = select_training_pairs(pairs, "train", 42, args.limit_per_family)
    checkpoint_path = Path("checkpoints") / args.run / "best.pt"
    digest = sha256(checkpoint_path)
    model, checkpoint = evaluation.load_checkpoint(checkpoint_path, torch.device("cuda"))
    evaluation.configure_precision(dict(strict_fp32=True))
    assert checkpoint["config"]["bn_policy"] == "batch_stats"
    assert checkpoint["config"]["fusion"] == "single_noex"
    model.eval()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad and name.startswith(("backbone.", "decode_head."))]
    scopes = {scope: [index for index, (name, _) in enumerate(parameters) if name.startswith(scope + ".")]
              for scope in ("backbone", "decode_head")}
    scopes["shared"] = list(range(len(parameters)))
    assert all(scopes.values())
    weight = checkpoint["config"]["mask_weight"]
    loader = DataLoader(PairDataset(selected, train=False, crop=None, jpeg=None), batch_size=1,
                        shuffle=False, num_workers=2)
    summary = dict(complete=False, checkpoint_sha256=digest, checkpoint_epoch=checkpoint["epoch"],
                   script_sha256=sha256(__file__), split_sha256=sha256(Path(args.data_root) / "splits/protocol_a_it0.csv"),
                   role="train", selected_ids=[pair["image_id"] for pair in selected],
                   mask_weight=weight, model_mode="eval", batch_size=1, augmentation=False,
                   strict_fp32=True, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32, optimizer_steps=0,
                   interpretation="Fixed-point per-image gradients, not training dynamics or causal proof.")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    with (output_dir / "per_image.csv").open("w", newline="") as stream:
        writer = None
        for index, batch in enumerate(loader):
            output = model(batch["x"].cuda())
            labels = batch["label"].cuda().float()
            image_loss = functional.binary_cross_entropy_with_logits(output["logit"].reshape(-1), labels.reshape(-1))
            total_loss, _ = referral_loss(output["logit"], output["map"], labels,
                                           batch["mask"].cuda(), batch["valid"].cuda(), mask_weight=weight)
            localization_loss = total_loss - image_loss
            image_gradients = torch.autograd.grad(image_loss, [parameter for _, parameter in parameters],
                                                  retain_graph=True, allow_unused=True)
            localization_gradients = torch.autograd.grad(localization_loss, [parameter for _, parameter in parameters],
                                                         allow_unused=True)
            pair = selected[index]
            assert batch["meta"]["image_id"][0] == pair["image_id"]
            row = dict(image_id=pair["image_id"], design_id=pair["design_id"],
                       family=pair["attack_family"] or "genuine", label=pair["label"],
                       logit_fake=float(output["logit"].detach().item()), image_loss=float(image_loss.detach()),
                       weighted_localization_loss=float(localization_loss.detach()))
            for scope, indices in scopes.items():
                geometry = gradient_geometry([image_gradients[position] for position in indices],
                                             [localization_gradients[position] for position in indices])
                row.update({f"{scope}_{name}": value for name, value in geometry.items()})
            assert all(parameter.grad is None for parameter in model.parameters())
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if (index + 1) % 24 == 0:
                print(f"scored {index + 1}/{len(selected)}", flush=True)
            del output, image_gradients, localization_gradients, total_loss, image_loss, localization_loss
    assert all(torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items())
    assert sha256(checkpoint_path) == digest
    summary.update(complete=True, model_state_unchanged=True, families=summarize_rows(rows))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["families"], indent=2), flush=True)


if __name__ == "__main__":
    main()
