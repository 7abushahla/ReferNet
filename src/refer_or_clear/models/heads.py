"""Decoder heads: MapHead (stride-4 alteration map logit) and ReferralHead (one image logit)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _conv_bn_relu(cin, cout, k=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MapHead(nn.Module):
    """Per stage 1x1 to mid, bilinear to the stride-4 size, concat, 3x3 to mid, 1x1 to one logit."""

    def __init__(self, in_channels_per_stage, mid=128):
        super().__init__()
        self.lateral = nn.ModuleList([_conv_bn_relu(c, mid) for c in in_channels_per_stage])
        self.merge = _conv_bn_relu(mid * len(self.lateral), mid, k=3)
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, feats):
        size = feats[0].shape[-2:]
        ups = [F.interpolate(l(f), size=size, mode="bilinear", align_corners=False) if f.shape[-2:] != size else l(f)
               for l, f in zip(self.lateral, feats)]
        return self.out(self.merge(torch.cat(ups, dim=1)))


class ReferralHead(nn.Module):
    """GAP of the deepest fused feature, concat max of the map logit, Linear-ReLU-Dropout-Linear."""

    def __init__(self, in_channels_deepest, hidden=256, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels_deepest + 1, hidden), nn.ReLU(inplace=True),
                                nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, deepest, map_logit):
        pooled = deepest.mean(dim=(2, 3))
        peak = map_logit.amax(dim=(1, 2, 3)).unsqueeze(1)
        return self.fc(torch.cat([pooled, peak], dim=1)).squeeze(1)
