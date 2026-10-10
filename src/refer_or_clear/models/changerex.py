"""ChangerEx arm: Open-CD IA_ResNetV1c-18 with Spatial/Channel Exchange and the Changer head (FDAF +
MixFFN), wrapped as a plain nn.Module with our (B, 8, H, W) -> {"logit", "map"} interface.

No mmengine Runner, no registered datasets: the two Open-CD modules are instantiated directly with the
arguments of configs/_base_/models/changer_r18.py and changer_ex_r18_512x512_40k_levircd.py (norm BN
instead of SyncBN, see evidence/code-audit-opencd-2026-10-10.md A.1). `models/build.py` should dispatch
arch == "changerex_r18" to `build_changerex(fusion, init, in_channels)`.

fusion  "difference": stock ChangerEx, query and reference through the shared backbone, FDAF fusion.
        "single":     matched single-image control: the reference branch receives a content-free
                      canvas (RGB zeros, i.e. the ImageNet mean after normalization) carrying the
                      query's own zone channel; x[:, 4:] is never read. Feeding the query to both
                      branches would be wrong here: FDAF fuses warp(x0) - x1 and warp(x1) - x0, so
                      identical branches give an all-zero fused feature and an untrainable head.
init    "scratch":    Open-CD's own init (Kaiming convs, zero-init residual BN, N(0, 0.01) conv_seg).
        "levir":      LEVIR-CD checkpoint (audit A.3); stem.0.weight zero-padded (32,3,3,3) ->
                      (32,in_channels,3,3); the 2-class conv_seg folded to one logit as w[1]-w[0],
                      b[1]-b[0], which equals logit[:,1]-logit[:,0] of the released head exactly.
"""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from opencd.models.backbones.interaction_resnet import IA_ResNetV1c
    from opencd.models.decode_heads.changer import Changer
    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001  (opencd asserts versions at import time)
    IA_ResNetV1c = Changer = None
    _IMPORT_ERROR = e

LEVIR_CKPT_NAME = "ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth"
LEVIR_CKPT_BYTES = 136_880_383
LEVIR_CKPT_MD5 = "7cd73048926efa8f0535317f5dde87e8"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACK_HINT = (
    "Open-CD stack not importable ({err!r}). Build it as in scripts/cluster/setup-env.sh: torch 2.1.2 + "
    "torchvision 0.16.2 (cu121 index), then `pip install mmcv==2.1.0 -f "
    "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html`, then mmengine==0.10.4 "
    "mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0 ftfy regex, then `pip install -e .` in a clone "
    "of https://github.com/likyoo/open-cd (evidence/code-audit-opencd-2026-10-10.md A.4; "
    "docs/implementation-plan.md Section 9 for the tested recipe)."
)

# changer_ex_r18_512x512_40k_levircd.py: identity after layer1, spatial exchange after layer2,
# channel exchange after layer3 and layer4 (all parameter-free).
INTERACTION_CFG = (
    None,
    dict(type="SpatialExchange", p=1 / 2),
    dict(type="ChannelExchange", p=1 / 2),
    dict(type="ChannelExchange", p=1 / 2),
)


def _require_opencd():
    if _IMPORT_ERROR is not None:
        raise ImportError(_STACK_HINT.format(err=_IMPORT_ERROR)) from _IMPORT_ERROR


@contextmanager
def _quiet():
    """Mute mmengine's per-parameter init_weights INFO log and mmseg's build_loss deprecation warnings."""
    from mmengine.logging import MMLogger  # init_weights logs via MMLogger.get_current_instance()
    lg = MMLogger.get_current_instance()
    prev = lg.level
    lg.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield
    finally:
        lg.setLevel(prev)


def _make_head_class():
    class ChangerWithFeat(Changer):
        """Changer head that also returns the 128-channel fused feature before cls_seg."""

        def forward(self, inputs):
            inputs = self._transform_inputs(inputs)
            halves = [torch.chunk(t, 2, dim=1) for t in inputs]
            out1 = self.base_forward([h[0] for h in halves])
            out2 = self.base_forward([h[1] for h in halves])
            feat = self.discriminator(self.neck_layer(out1, out2, "concat"))
            return feat, self.cls_seg(feat)

    return ChangerWithFeat


class ReferralHead(nn.Module):
    """GAP of the fused feature concatenated with max of the map logit, two dense layers, one logit."""

    def __init__(self, in_channels=128, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels + 1, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, feat, map_logit):
        pooled = feat.mean(dim=(2, 3))
        peak = map_logit.flatten(1).amax(dim=1, keepdim=True)
        return self.fc2(F.relu(self.fc1(torch.cat([pooled, peak], dim=1)))).squeeze(1)


class ChangerEx(nn.Module):
    """Siamese ChangerEx r18 on [query RGB, zone | reference RGB, zone] with a referral head."""

    def __init__(self, fusion="difference", in_channels=4, norm_cfg=None):
        super().__init__()
        _require_opencd()
        if fusion not in ("difference", "single"):
            raise ValueError(f"changerex_r18 supports fusion in {{'difference', 'single'}}, got {fusion!r}")
        self.fusion = fusion
        self.in_channels = in_channels
        norm_cfg = norm_cfg or dict(type="BN", requires_grad=True)
        with _quiet():
            self.backbone = IA_ResNetV1c(
                depth=18, in_channels=in_channels, interaction_cfg=INTERACTION_CFG, num_stages=4,
                out_indices=(0, 1, 2, 3), dilations=(1, 1, 1, 1), strides=(1, 2, 2, 2), norm_cfg=norm_cfg,
                norm_eval=False, style="pytorch", contract_dilation=True)
            self.decode_head = _make_head_class()(
                in_channels=[64, 128, 256, 512], in_index=[0, 1, 2, 3], channels=128, dropout_ratio=0.1,
                num_classes=2, out_channels=1, threshold=0.5, norm_cfg=norm_cfg, align_corners=False)
        self.referral = ReferralHead(in_channels=128)

    def init_opencd_weights(self):
        """Apply Open-CD's init_cfg (mmengine does this inside the Runner; we have none)."""
        with _quiet():
            self.backbone.init_weights()
            self.decode_head.init_weights()

    def split(self, x):
        c = self.in_channels
        query = x[:, :c]
        if self.fusion == "single":
            ref = torch.cat([torch.zeros_like(query[:, :c - 1]), query[:, c - 1:c]], dim=1)
        else:
            ref = x[:, c:2 * c]
        return query, ref

    def forward(self, x):
        query, ref = self.split(x)
        feats = self.backbone(query, ref)  # per stage: cat([f_query, f_ref]) at strides 4, 8, 16, 32
        feat, map_logit = self.decode_head(feats)  # (B,128,H/4,W/4), (B,1,H/4,W/4)
        return {"logit": self.referral(feat, map_logit), "map": map_logit}


def default_levir_ckpt():
    return Path(os.environ.get("ROC_LEVIR_CKPT", _REPO_ROOT / "checkpoints" / "pretrained" / LEVIR_CKPT_NAME))


def load_levir(model, ckpt_path=None, verbose=True):
    """Load the LEVIR ChangerEx state_dict with the stem pad and the conv_seg fold; returns the
    (missing, unexpected) key lists."""
    path = Path(ckpt_path) if ckpt_path else default_levir_ckpt()
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/cluster/get_levir_ckpt.sh")
    ck = torch.load(path, map_location="cpu")
    sd = dict(ck.get("state_dict", ck))
    w = sd["backbone.stem.0.weight"]
    c = model.in_channels
    if w.shape[1] != c:
        padded = w.new_zeros((w.shape[0], c) + tuple(w.shape[2:]))
        padded[:, :min(c, w.shape[1])] = w[:, :min(c, w.shape[1])]
        sd["backbone.stem.0.weight"] = padded
    cw, cb = sd["decode_head.conv_seg.weight"], sd["decode_head.conv_seg.bias"]
    if cw.shape[0] == 2:  # (unchanged, changed) softmax pair -> one "changed" log-odds logit
        sd["decode_head.conv_seg.weight"] = cw[1:] - cw[0:1]
        sd["decode_head.conv_seg.bias"] = cb[1:] - cb[0:1]
    res = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[changerex] loaded {path.name}: {len(sd) - len(res.unexpected_keys)} tensors matched, "
              f"stem {tuple(w.shape)} -> {tuple(sd['backbone.stem.0.weight'].shape)}, "
              f"conv_seg {tuple(cw.shape)} -> {tuple(sd['decode_head.conv_seg.weight'].shape)}")
        print(f"[changerex] missing ({len(res.missing_keys)}): {res.missing_keys}")
        print(f"[changerex] unexpected ({len(res.unexpected_keys)}): {res.unexpected_keys}")
    return res.missing_keys, res.unexpected_keys


def build_changerex(fusion="difference", init="levir", in_channels=4, ckpt_path=None) -> nn.Module:
    """Entry point for build.py (arch == "changerex_r18"). init in {"scratch", "levir"}."""
    model = ChangerEx(fusion=fusion, in_channels=in_channels)
    model.init_opencd_weights()
    if init == "levir":
        load_levir(model, ckpt_path)
    elif init != "scratch":
        raise ValueError(f"changerex_r18 supports init in {{'scratch', 'levir'}}, got {init!r}")
    return model
