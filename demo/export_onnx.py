# demo/export_onnx.py
"""Export the fold-0 deployment pair (ReferNet OFF and ON) to ONNX for the demo website.

The wrappers re-express DirectReferenceClassifier.forward for one document whose reference context is available
(the site never runs the network without it), without the data-dependent branches that ONNX tracing cannot keep.
ON also returns the comparison features before pooling, from which the site draws the difference map.

Usage: python demo/export_onnx.py [--ckpt demo/ckpt] [--out site/model]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from bench_cpu_deploy import load_arm  # noqa: E402
from refer_or_clear.models.anchor_changer import encode_query  # noqa: E402


def _masked_pool(features, mask):
    """_LocalAdapter._masked_pool with a float zero: its torch.where(..., maximum, 0) exports the 0 as int64,
    which ONNX Runtime rejects (Where bound to float and int64)."""
    count = mask.sum(dim=(2, 3)).clamp_min(1)
    mean = features.masked_fill(~mask, 0).sum(dim=(2, 3)) / count
    maximum = features.masked_fill(~mask, -torch.inf).amax(dim=(2, 3))
    maximum = torch.where(mask.flatten(1).any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


class OnExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model.reference_enabled and model.local_enabled and model.fused_query_weight == 1.0
        self.m = model

    def forward(self, query, reference, overlap):
        m, a = self.m, self.m.adapter
        stages = encode_query(m.base.backbone, query)
        query_logit = m.base.classifier(stages[-1].mean(dim=(2, 3))).squeeze(1)
        mask = F.interpolate(overlap, size=(query.shape[2] // 8, query.shape[3] // 8), mode="nearest") > 0.5
        pq = a.projection(stages[1])
        pr = a.projection(m._reference_prefix(torch.cat((reference, query[:, 3:4]), dim=1)))
        diff = (pq - pr).abs()
        zone = F.interpolate(query[:, 3:4].clamp(0, 1), size=pq.shape[2:], mode="nearest")
        comparison = torch.cat((pq.masked_fill(~mask, 0), pr.masked_fill(~mask, 0), diff.masked_fill(~mask, 0),
                                zone.masked_fill(~mask, 0), mask.to(pq.dtype)), dim=1)
        local = a.local_head(comparison)
        correction = a.correction_head(_masked_pool(local, mask)).squeeze(1)
        return query_logit, correction, local


class OffExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert not model.local_enabled
        self.m = model

    def forward(self, query):
        stages = encode_query(self.m.base.backbone, query)
        return self.m.base.classifier(stages[-1].mean(dim=(2, 3))).squeeze(1)


def export(ckpt_dir: Path, out_dir: Path) -> dict:
    ckpt_dir, out_dir = Path(ckpt_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "weights.pth"
    on, _ = load_arm(ckpt_dir / "on/best.pt", weights, "on")
    off, _ = load_arm(ckpt_dir / "off/best.pt", weights, "off")
    q, r, ov = torch.zeros(1, 4, 512, 800), torch.zeros(1, 3, 512, 800), torch.ones(1, 1, 512, 800)
    with torch.no_grad():
        torch.onnx.export(OnExport(on), (q, r, ov), str(out_dir / "refernet_on.onnx"), opset_version=17,
                          input_names=["query", "reference", "overlap"],
                          output_names=["query_logit", "correction", "local"], do_constant_folding=True)
        torch.onnx.export(OffExport(off), (q,), str(out_dir / "refernet_off.onnx"), opset_version=17,
                          input_names=["query"], output_names=["query_logit"], do_constant_folding=True)
    w = on.adapter.correction_head.weight.detach().flatten().tolist()
    head = {"w_mean": w[:32], "w_max": w[32:], "bias": float(on.adapter.correction_head.bias.item())}
    (out_dir / "head.json").write_text(json.dumps(head))
    return {"head": head, "files": sorted(p.name for p in out_dir.iterdir())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "demo/ckpt")); ap.add_argument("--out", default=str(ROOT / "site/model"))
    a = ap.parse_args()
    print(json.dumps(export(Path(a.ckpt), Path(a.out))["files"]))
