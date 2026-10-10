"""build_model(arch, fusion, init, in_channels) -> nn.Module (docs/implementation-plan.md, Section 3).

Input (B, 8, H, W) = [query RGB, query zone, reference RGB, reference zone]; output
{"logit": (B,), "map": (B, 1, H/4, W/4)}. The encoder is a torchvision ResNet-18 with the stem
widened to in_channels (pretrained RGB filters copied, extra channels zero-initialised), shared by
the query and reference branches. 'stacked' runs one encoder over all 2 * in_channels channels.
"""
from __future__ import annotations

import warnings

import torch
from torch import nn
import torchvision

from .fusion import KINDS, build_fusion
from .heads import MapHead, ReferralHead

STAGE_CHANNELS = (64, 128, 256, 512)  # strides 4, 8, 16, 32


def _resnet18(init):
    if init == "scratch":
        return torchvision.models.resnet18(weights=None)
    if init != "imagenet":
        raise ValueError(f"unknown init {init!r}; expected 'imagenet' or 'scratch'")
    try:
        return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:  # offline or no cache: keep going so smoke tests run anywhere
        warnings.warn(f"ImageNet weights unavailable ({type(e).__name__}: {e}); using random init")
        return torchvision.models.resnet18(weights=None)


def _widen_stem(conv, in_channels):
    """New stem conv: the first min(3, in_channels) filters are copied, the rest are zero."""
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        new.weight.zero_()
        k = min(conv.in_channels, in_channels)
        new.weight[:, :k] = conv.weight[:, :k]
    return new


class ResNet18Encoder(nn.Module):
    """ResNet-18 trunk returning features at strides 4, 8, 16, 32 (64, 128, 256, 512 channels)."""

    def __init__(self, init="imagenet", in_channels=4):
        super().__init__()
        r = _resnet18(init)
        self.stem = nn.Sequential(_widen_stem(r.conv1, in_channels), r.bn1, r.relu, r.maxpool)
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.in_channels, self.out_channels = in_channels, list(STAGE_CHANNELS)

    def forward(self, x):
        x, feats = self.stem(x), []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class SiameseNet(nn.Module):
    def __init__(self, fusion="difference", init="imagenet", in_channels=4):
        super().__init__()
        if fusion not in KINDS:
            raise ValueError(f"unknown fusion {fusion!r}; expected one of {KINDS}")
        self.fusion_kind, self.in_channels = fusion, in_channels
        self.encoder = ResNet18Encoder(init, 2 * in_channels if fusion == "stacked" else in_channels)
        self.fusion = build_fusion(fusion, self.encoder.out_channels)
        self.map_head = MapHead(self.fusion.out_channels)
        self.referral_head = ReferralHead(self.fusion.out_channels[-1])

    def parameter_groups(self):
        """(pretrained encoder params, new params) for the two learning rates of train.py."""
        enc = set(id(p) for p in self.encoder.parameters())
        return list(self.encoder.parameters()), [p for p in self.parameters() if id(p) not in enc]

    def forward(self, x):
        c = self.in_channels
        if self.fusion_kind == "stacked":
            fused = self.fusion(self.encoder(x))
        elif self.fusion_kind == "single":
            fused = self.fusion(self.encoder(x[:, :c]))
        else:
            fused = self.fusion(self.encoder(x[:, :c]), self.encoder(x[:, c:2 * c]))
        m = self.map_head(fused)
        return {"logit": self.referral_head(fused[-1], m), "map": m}


def build_model(arch="siamese_r18", fusion="difference", init="imagenet", in_channels=4) -> nn.Module:
    if arch == "siamese_r18":
        return SiameseNet(fusion=fusion, init=init, in_channels=in_channels)
    if arch == "changerex_r18":
        from .changerex import build_changerex
        return build_changerex(fusion, init, in_channels=in_channels)
    raise ValueError(f"unknown arch {arch!r}; expected 'siamese_r18' or 'changerex_r18'")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
