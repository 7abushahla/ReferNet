#!/usr/bin/env python3
"""Train-only branch-specific BN calibration on frozen checkpoints, never test data.

This is a diagnostic intervention, not a production normalization policy. Calibration
collects population moments under batch-normalized activations. Deeper-layer input
distributions may change when those moments are frozen for inference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class BranchMoments(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.collecting = True
        self.calls = 0
        self.expected_calls = None
        self.moments = []

    def forward(self, inputs):
        branch = self.calls
        self.calls += 1
        if self.collecting:
            if branch == len(self.moments):
                self.moments.append(dict(count=0, total=0, squares=0))
            block = self.moments[branch]
            variance, mean = torch.var_mean(inputs.detach().float(), dim=(0, 2, 3), unbiased=False)
            mean, variance = mean.double(), variance.double()
            count = inputs.numel() // inputs.shape[1]
            block["count"] += count
            block["total"] = block["total"] + mean * count
            block["squares"] = block["squares"] + (variance + mean.square()) * count
            return functional.batch_norm(inputs, None, None, self.weight, self.bias, True, 0., self.eps)
        block = self.moments[branch]
        return functional.batch_norm(inputs, block["mean"], block["variance"],
                                     self.weight, self.bias, False, 0., self.eps)

    def freeze(self):
        if self.expected_calls not in (1, 2):
            raise ValueError(f"expected one or two BN calls, got {self.expected_calls}")
        self.collecting = False
        for block in self.moments:
            mean = block["total"] / block["count"]
            variance = (block["squares"] / block["count"] - mean.square()).clamp_min(0)
            block["mean"], block["variance"] = mean.float(), variance.float()
        return dict(calls_per_forward=self.expected_calls,
                    mean=[block["mean"].cpu().tolist() for block in self.moments],
                    variance=[block["variance"].cpu().tolist() for block in self.moments],
                    count=[block["count"] for block in self.moments])


@contextmanager
def calibrated_branches(model, loader, device, role, scope):
    if role != "train":
        raise ValueError("calibration requires training data")
    if scope not in ("backbone", "all"):
        raise ValueError(scope)
    settings = [(module, module.training) for module in model.modules()]
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, nn.BatchNorm2d) and (scope == "all" or name.startswith("backbone."))]
    replacements = []
    handles = []
    try:
        model.eval()
        for name, original in selected:
            parent_path, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            replacement = BranchMoments(original)
            replacements.append((name, parent, child, original, replacement))
            setattr(parent, child, replacement)

        def begin_forward(module, inputs):
            for _, _, _, _, replacement in replacements:
                replacement.calls = 0

        def end_forward(module, inputs, outputs):
            for name, _, _, _, replacement in replacements:
                if replacement.expected_calls is None:
                    replacement.expected_calls = replacement.calls
                if replacement.calls != replacement.expected_calls:
                    raise AssertionError(f"branch call count changed at {name}")

        handles = [model.register_forward_pre_hook(begin_forward), model.register_forward_hook(end_forward)]
        with torch.no_grad():
            for batch in loader:
                model(batch["x"].to(device))
        report = {name: replacement.freeze() for name, _, _, _, replacement in replacements}
        yield report
    finally:
        for handle in handles:
            handle.remove()
        for _, parent, child, original, _ in replacements:
            setattr(parent, child, original)
        for module, training in settings:
            module.training = training


def main():
    import numpy as np
    from refer_or_clear import evaluate as evaluation
    from diagnose_anchor_gap import dump, group_summary, score_ours, write_rows
    from diagnose_control import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="A-cex-single_noex-median-levir-f0-s42-bnbatch-e100")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    info = json.loads((ROOT / "results/runs" / args.run / "train.json").read_text())
    config = info["raw_config"]
    split, records = evaluation.load_split(config, args.data_root, fold=0)
    pairs = {role: evaluation.role_pairs(split, records, args.data_root, role, "median", config, 42,
                                         wrong_design_fraction=0.0) for role in ("train", "val")}
    assert not ({pair["image_id"] for pair in pairs["train"]} & {pair["image_id"] for pair in pairs["val"]})
    np.random.RandomState(42).shuffle(pairs["train"])
    calibration_loader = evaluation.eval_loader(pairs["train"], 8, 2)
    path = ROOT / "checkpoints" / args.run / "best.pt"
    digest = sha256(path)
    model, checkpoint = evaluation.load_checkpoint(path, device)
    assert checkpoint["config"]["fold"] == 0 and checkpoint["config"]["protocol"] == "A"
    assert checkpoint["config"]["fusion"] == "single_noex"
    assert checkpoint["config"]["bn_policy"] == "batch_stats"
    model.eval()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    summary = dict(run=args.run, checkpoint_sha256=digest, epoch=checkpoint["epoch"],
                   script_sha256=sha256(__file__), test_used=False, complete=False,
                   calibration="all native training images, batch 8; population moments, branch separated",
                   scope="fold 0 selected checkpoint only; all probes on validation", results={})
    dump(out / "inputs.json", {role: [pair["image_id"] for pair in selected] for role, selected in pairs.items()})
    dump(out / "train_metadata.json", info)
    rows = score_ours(model, pairs["val"], "native", 1, device)
    summary["results"]["baseline"] = group_summary(rows)
    write_rows(out / "baseline_val.csv", rows)
    dump(out / "summary.json", summary)
    for scope in ("backbone", "all"):
        with calibrated_branches(model, calibration_loader, device, "train", scope) as moments:
            dump(out / f"{scope}_moments.json", moments)
            rows = score_ours(model, pairs["val"], "native", 1, device)
            summary["results"][scope] = group_summary(rows)
            write_rows(out / f"{scope}_val.csv", rows)
            together = score_ours(model, pairs["val"][:16], "native", 8, device)
            individual = torch.tensor([row["logit_fake"] for row in rows[:16]])
            batched = torch.tensor([row["logit_fake"] for row in together])
            torch.testing.assert_close(individual, batched, rtol=1e-4, atol=1e-4)
            summary["results"][scope]["batch_invariance_max_logit_difference"] = float((individual - batched).abs().max())
        assert all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
        print(scope, {name: block["auc_pipeline_score"] for name, block in summary["results"][scope].items()
                      if isinstance(block, dict)}, flush=True)
        dump(out / "summary.json", summary)
    assert sha256(path) == digest
    summary["complete"] = True
    dump(out / "summary.json", summary)


if __name__ == "__main__":
    main()
