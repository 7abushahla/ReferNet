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
        "attention_single": the improved variant's matched control (attention modules, content-free reference).
        "single_noex": the fair single-image control (11 Oct 2026): content-free reference AND the exchange
                      layers switched off, so the query keeps all its features. With "single" the spatial and
                      channel exchanges still swap half of the query's features for the blank branch's at three
                      depths, which halves the signal; Protocol A measured it at chance (AUC 0.60 on seen designs).
                      Same backbone, weights, heads and parameter count as "difference".
        "attention":  the improved variant, the dashed boxes of the proposal figure: at strides 16 and
                      32 the query features pass a pre-LN self-attention block and then a cross-
                      attention block (Q = f(x), K = V = f(r)), each residual on f(x), before the
                      stock decode head (FDAF untouched); in the referral path the pooled global
                      descriptor (GAP of the fused feature with max of the map) attends once over
                      the fused local tokens ("global over local") before the two dense layers.
                      Every output projection is zero-initialised, so at step 0 the model computes
                      exactly what "difference" computes.
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

    def descriptor(self, feat, map_logit):
        pooled = feat.mean(dim=(2, 3))
        peak = map_logit.flatten(1).amax(dim=1, keepdim=True)
        return torch.cat([pooled, peak], dim=1)

    def classify(self, g):
        return self.fc2(F.relu(self.fc1(g))).squeeze(1)

    def forward(self, feat, map_logit):
        return self.classify(self.descriptor(feat, map_logit))


def _zero(linear):
    nn.init.zeros_(linear.weight), nn.init.zeros_(linear.bias)
    return linear


class _Attention(nn.Module):
    """Pre-LN multi-head attention between token sets: out = W_o MHA(W_q LN(q), W_kv LN(kv), W_kv LN(kv)),
    with W_o zero-initialised so that `x + out` starts as the identity on x."""

    def __init__(self, c_q, c_kv, c_out, heads=6, key_dim=32):
        super().__init__()
        d = heads * key_dim
        self.norm_q, self.norm_kv = nn.LayerNorm(c_q), nn.LayerNorm(c_kv)
        self.to_q, self.to_kv = nn.Linear(c_q, d), nn.Linear(c_kv, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.to_out = _zero(nn.Linear(d, c_out))

    def forward(self, q, kv):
        # Attention runs in float32 even under autocast: the softmax over 1,600 tokens overflowed in
        # float16 and produced NaN losses in the first campaign (10 Oct, jobs 37671 and 37672).
        with torch.autocast(q.device.type, enabled=False):
            q, kv = q.float(), kv.float()
            kv = self.to_kv(self.norm_kv(kv))
            return self.to_out(self.attn(self.to_q(self.norm_q(q)), kv, kv, need_weights=False)[0])


class QueryAttention(nn.Module):
    """One backbone stage of the improved variant: self-attention over the f(x) tokens, then cross-
    attention with Q = f(x), K = V = f(r); both residual on f(x). Returns f(x) with the map shape."""

    def __init__(self, c, heads=6, key_dim=32):
        super().__init__()
        self.self_attn = _Attention(c, c, c, heads, key_dim)
        self.cross_attn = _Attention(c, c, c, heads, key_dim)

    def forward(self, q, r):
        b, c, h, w = q.shape
        tq, tr = q.flatten(2).transpose(1, 2), r.flatten(2).transpose(1, 2)
        tq = tq + self.self_attn(tq, tq)
        tq = tq + self.cross_attn(tq, tr)
        return tq.transpose(1, 2).reshape(b, c, h, w)


class GlobalOverLocal(nn.Module):
    """Referral-path cross-attention: the 129-d global descriptor (one token) attends over the fused
    stride-4 feature, average-pooled `pool` x `pool` first (32 x 50 = 1,600 tokens at 512 x 800);
    residual onto the descriptor, which then enters the two dense layers unchanged in width."""

    def __init__(self, c_global=129, c_local=128, pool=4, heads=6, key_dim=32):
        super().__init__()
        self.pool = pool
        self.attn = _Attention(c_global, c_local, c_global, heads, key_dim)

    def forward(self, g, feat):
        tokens = F.avg_pool2d(feat, self.pool).flatten(2).transpose(1, 2)
        return g + self.attn(g.unsqueeze(1), tokens).squeeze(1)


class ChangerEx(nn.Module):
    """Siamese ChangerEx r18 on [query RGB, zone | reference RGB, zone] with a referral head."""

    def __init__(self, fusion="difference", in_channels=4, norm_cfg=None):
        super().__init__()
        _require_opencd()
        if fusion not in ("difference", "single", "single_noex", "attention", "attention_single"):
            raise ValueError(f"changerex_r18 supports fusion in {{'difference', 'single', 'single_noex', "
                             f"'attention', 'attention_single'}}, got {fusion!r}")
        self.fusion = fusion
        self.in_channels = in_channels
        norm_cfg = norm_cfg or dict(type="BN", requires_grad=True)
        with _quiet():
            self.backbone = IA_ResNetV1c(
                depth=18, in_channels=in_channels,
                interaction_cfg=(None, None, None, None) if fusion == "single_noex" else INTERACTION_CFG, num_stages=4,
                out_indices=(0, 1, 2, 3), dilations=(1, 1, 1, 1), strides=(1, 2, 2, 2), norm_cfg=norm_cfg,
                norm_eval=False, style="pytorch", contract_dilation=True)
            self.decode_head = _make_head_class()(
                in_channels=[64, 128, 256, 512], in_index=[0, 1, 2, 3], channels=128, dropout_ratio=0.1,
                num_classes=2, out_channels=1, threshold=0.5, norm_cfg=norm_cfg, align_corners=False)
        self.referral = ReferralHead(in_channels=128)
        self.attention = None
        if fusion in ("attention", "attention_single"):
            # Built on a forked RNG stream so that the backbone, head and referral tensors receive the same
            # draws as in the other variants (the test relies on it; the initial behaviour then matches).
            with torch.random.fork_rng(devices=[]):
                self.attention = nn.ModuleDict({"stage16": QueryAttention(256), "stage32": QueryAttention(512),
                                                "global": GlobalOverLocal(129, 128)})

    def init_opencd_weights(self):
        """Apply Open-CD's init_cfg (mmengine does this inside the Runner; we have none)."""
        with _quiet():
            self.backbone.init_weights()
            self.decode_head.init_weights()

    def split(self, x):
        c = self.in_channels
        query = x[:, :c]
        if self.fusion in ("single", "single_noex", "attention_single"):
            ref = torch.cat([torch.zeros_like(query[:, :c - 1]), query[:, c - 1:c]], dim=1)
        else:
            ref = x[:, c:2 * c]
        return query, ref

    def parameter_groups(self):
        """(pretrained params, new params) for the two learning rates: backbone and decode head come
        from LEVIR (or Open-CD init), the referral head and the attention modules are ours. The zone
        channel of the stem is part of the pretrained tensor and follows the backbone rate."""
        pre = list(self.backbone.parameters()) + list(self.decode_head.parameters())
        new = list(self.referral.parameters())
        if self.attention is not None:   # attention modules take the lower (backbone) rate: at lr_new the
            pre += list(self.attention.parameters())   # validation loss spiked and two runs diverged
        return pre, new

    def forward(self, x):
        query, ref = self.split(x)
        feats = list(self.backbone(query, ref))  # per stage: cat([f_query, f_ref]) at strides 4, 8, 16, 32
        if self.attention is not None:
            for i, key in ((2, "stage16"), (3, "stage32")):
                f_q, f_r = torch.chunk(feats[i], 2, dim=1)
                feats[i] = torch.cat([self.attention[key](f_q, f_r), f_r], dim=1)
        feat, map_logit = self.decode_head(feats)  # (B,128,H/4,W/4), (B,1,H/4,W/4)
        g = self.referral.descriptor(feat, map_logit)
        if self.attention is not None:
            g = self.attention["global"](g, feat)
        return {"logit": self.referral.classify(g), "map": map_logit}


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


def set_encoder_bn_policy(model, policy):
    if policy not in ("running", "batch_stats"):
        raise ValueError(f"unknown encoder BN policy {policy!r}")
    if policy == "batch_stats":
        for module in model.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None


def build_changerex(fusion="difference", init="levir", in_channels=4, ckpt_path=None,
                    bn_policy="running") -> nn.Module:
    """Entry point for build.py (arch == "changerex_r18"). init in {"scratch", "levir"}."""
    model = ChangerEx(fusion=fusion, in_channels=in_channels)
    model.init_opencd_weights()
    if init == "levir":
        load_levir(model, ckpt_path)
    elif init != "scratch":
        raise ValueError(f"changerex_r18 supports init in {{'scratch', 'levir'}}, got {init!r}")
    set_encoder_bn_policy(model, bn_policy)
    return model
