"""Fusion of query and reference features (docs/implementation-plan.md, Section 3).

single     : query features only; the reference is ignored (same-capacity baseline).
stacked    : no feature fusion; build_model runs one encoder on the 8-channel tensor and the
             fusion module is the identity on that encoder's features.
difference : per stage concat(f_q, |f_q - f_r|) -> 1x1 conv + BN + ReLU back to the stage width.
attention  : strides 4 and 8 use the difference fusion; strides 16 and 32 apply multi-head
             cross-attention (queries from f_q, keys and values from f_r, 6 heads of key dim 32)
             with a residual add onto f_q. The additive zone bias on the attention logits named in
             the plan is not implemented here: the fusion sees features only, and the zone already
             enters both branches through the stem's fourth channel.

build_fusion(kind, channels_per_stage) returns a module with forward(f_q: list, f_r: list) -> list
and an attribute out_channels (list, one entry per stage).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

KINDS = ("single", "stacked", "difference", "attention")


def _conv_bn_relu(cin, cout, k=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SingleFusion(nn.Module):
    """Identity on the query features (used by 'single' and 'stacked')."""

    def __init__(self, channels):
        super().__init__()
        self.out_channels = list(channels)

    def forward(self, f_q, f_r=None):
        return list(f_q)


class DifferenceStage(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = _conv_bn_relu(2 * c, c)

    def forward(self, q, r):
        return self.proj(torch.cat([q, (q - r).abs()], dim=1))


class CrossAttentionStage(nn.Module):
    """Cross-attention over flattened tokens: q <- q + W_o MHA(LN(q), LN(r), LN(r))."""

    def __init__(self, c, heads=6, key_dim=32, dropout=0.0):
        super().__init__()
        d = heads * key_dim
        self.norm_q, self.norm_r = nn.LayerNorm(c), nn.LayerNorm(c)
        self.to_q, self.to_kv = nn.Linear(c, d), nn.Linear(c, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.to_out = nn.Linear(d, c)
        nn.init.zeros_(self.to_out.weight), nn.init.zeros_(self.to_out.bias)  # start as identity on f_q

    def forward(self, q, r):
        b, c, h, w = q.shape
        tq = self.norm_q(q.flatten(2).transpose(1, 2))
        tr = self.norm_r(r.flatten(2).transpose(1, 2))
        kv = self.to_kv(tr)
        out, _ = self.attn(self.to_q(tq), kv, kv, need_weights=False)
        return q + self.to_out(out).transpose(1, 2).reshape(b, c, h, w)


class StagewiseFusion(nn.Module):
    def __init__(self, stages, channels):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(channels)

    def forward(self, f_q, f_r):
        return [m(q, r) for m, q, r in zip(self.stages, f_q, f_r)]


def build_fusion(kind, channels_per_stage):
    channels = list(channels_per_stage)
    if kind in ("single", "stacked"):
        return SingleFusion(channels)
    if kind == "difference":
        return StagewiseFusion([DifferenceStage(c) for c in channels], channels)
    if kind == "attention":
        n = len(channels)
        stages = [DifferenceStage(c) if i < n - 2 else CrossAttentionStage(c) for i, c in enumerate(channels)]
        return StagewiseFusion(stages, channels)
    raise ValueError(f"unknown fusion {kind!r}; expected one of {KINDS}")
