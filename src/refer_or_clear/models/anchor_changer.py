"""Isolated query-only direct and absent pilots using installed Open-CD Changer modules."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch import Tensor, nn

from refer_or_clear.anchor_contract import contract_for
from . import changerex


_NEW_HISTORICAL_KEYS = frozenset(
    f"referral.{layer}.{kind}" for layer in ("fc1", "fc2") for kind in ("weight", "bias")
)
_STEM_KEY = "backbone.stem.0.weight"
_MAP_WEIGHT = "decode_head.conv_seg.weight"
_MAP_BIAS = "decode_head.conv_seg.bias"
_TRUSTED_LEVIR_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"


@contextmanager
def _seeded_construction(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _verified_checkpoint(path: Path, expected_sha256: str) -> tuple[dict[str, Tensor], str]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("Expected checkpoint SHA256 must contain 64 hexadecimal characters")
    if any(character not in "0123456789abcdefABCDEF" for character in expected_sha256):
        raise ValueError("Expected checkpoint SHA256 must contain 64 hexadecimal characters")
    checkpoint_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"LEVIR checkpoint SHA256 mismatch for {path}")
    try:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu",
                                weights_only=actual_sha256 != _TRUSTED_LEVIR_SHA256)
    except pickle.UnpicklingError as error:
        raise ValueError("Unsupported checkpoint metadata; unrestricted loading requires the official LEVIR hash allowlist") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("LEVIR checkpoint must be a tensor state dictionary or contain state_dict")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(name, str) and isinstance(tensor, Tensor) for name, tensor in state.items()):
        raise ValueError("LEVIR state_dict must contain named tensors only")
    return dict(state), actual_sha256


def _tensor_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        descriptor = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":"))
        digest.update(descriptor.encode("utf-8") + b"\n")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_declared_levir(historical: nn.Module, source: dict[str, Tensor]) -> dict:
    target = historical.state_dict()
    if not _NEW_HISTORICAL_KEYS <= target.keys():
        raise ValueError("Historical referral head does not match the declared new-key allowlist")
    allowed = target.keys() - _NEW_HISTORICAL_KEYS
    missing = sorted(allowed - source.keys())
    unexpected = sorted(source.keys() - allowed)
    if missing or unexpected:
        raise ValueError(f"LEVIR tensor key coverage mismatch, missing keys {missing}, unexpected keys {unexpected}")
    transplanted = dict(source)
    stem = source[_STEM_KEY]
    expected_stem = target[_STEM_KEY]
    if stem.ndim != 4 or stem.shape[1] != 3 or expected_stem.shape[1] != 4 or stem.shape[0] != expected_stem.shape[0] or stem.shape[2:] != expected_stem.shape[2:]:
        raise ValueError(f"LEVIR stem shape must be RGB widened to four channels, got {tuple(stem.shape)}")
    padded = stem.new_zeros(expected_stem.shape)
    padded[:, :3] = stem
    transplanted[_STEM_KEY] = padded
    map_weight, map_bias = source[_MAP_WEIGHT], source[_MAP_BIAS]
    if map_weight.shape != (2, *target[_MAP_WEIGHT].shape[1:]) or map_bias.shape != (2,):
        raise ValueError("LEVIR map head must contain the original two-class conv_seg tensors")
    transplanted[_MAP_WEIGHT] = map_weight[1:] - map_weight[:1]
    transplanted[_MAP_BIAS] = map_bias[1:] - map_bias[:1]
    for name, tensor in transplanted.items():
        expected = target[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError(f"LEVIR tensor shape/dtype mismatch for {name}, got {tuple(tensor.shape)} {tensor.dtype}, expected {tuple(expected.shape)} {expected.dtype}")
    result = historical.load_state_dict(transplanted, strict=False)
    if set(result.missing_keys) != _NEW_HISTORICAL_KEYS or result.unexpected_keys:
        raise ValueError(f"LEVIR load returned unallowlisted keys, missing {result.missing_keys}, unexpected {result.unexpected_keys}")
    loaded_state = {name: historical.state_dict()[name] for name in transplanted}
    return {
        "loaded_keys": sorted(transplanted),
        "missing_keys": sorted(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "historical_pretrained_tensor_sha256": _tensor_sha256(loaded_state),
        "surgeries": {
            "stem_zero_padding": {"source_shape": list(stem.shape), "target_shape": list(padded.shape)},
            "conv_seg_log_odds": {"source_shape": list(map_weight.shape), "target_shape": list(transplanted[_MAP_WEIGHT].shape)},
        },
    }


def encode_query(backbone: nn.Module, query: Tensor) -> list[Tensor]:
    """Traverse IA_ResNetV1c's three-convolution stem and residual stages once."""
    if not backbone.deep_stem or tuple(backbone.res_layers) != ("layer1", "layer2", "layer3", "layer4"):
        raise ValueError("D/N require the declared IA_ResNetV1c deep stem and four residual stages")
    if tuple(backbone.out_indices) != (0, 1, 2, 3):
        raise ValueError("D/N require all four IA_ResNetV1c output stages")
    features = backbone.maxpool(backbone.stem(query))
    outputs = []
    for layer_name, width in zip(backbone.res_layers, (64, 128, 256, 512)):
        features = getattr(backbone, layer_name)(features)
        if features.shape[1] != width:
            raise ValueError(f"Unexpected query feature width at {layer_name}, expected {width}")
        outputs.append(features)
    return outputs


def _normalization_audit(model: nn.Module) -> list[dict]:
    audit = []
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm, nn.LayerNorm, nn.GroupNorm)
    for name, module in model.named_modules():
        if isinstance(module, norm_types):
            audit.append({"name": name, "type": type(module).__name__, "track_running_stats": getattr(module, "track_running_stats", None)})
        elif isinstance(module, nn.modules.dropout._DropoutNd) or hasattr(module, "drop_prob"):
            audit.append({"name": name, "type": type(module).__name__, "probability": float(getattr(module, "p", getattr(module, "drop_prob", 0.0)))})
    return audit


class AnchorChanger(nn.Module):
    def __init__(self, historical: nn.Module, track: str, head_seed: int):
        super().__init__()
        contract = contract_for(track)
        self.track = track
        self.architecture_id = contract.architecture_id
        self.bn_policy = contract.bn_policy
        self.backbone = historical.backbone
        if track in ("D", "D_running", "D_running_pilot"):
            with _seeded_construction(head_seed):
                self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
            changerex.set_encoder_bn_policy(self, self.bn_policy)
        else:
            self.decode_head = historical.decode_head
            self.referral = historical.referral
        self.initialization_provenance = {}
        self._check_bn_policy()

    def _check_bn_policy(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                running = self.bn_policy == "running"
                buffers_present = all(getattr(module, key) is not None for key in ("running_mean", "running_var", "num_batches_tracked"))
                buffers_absent = all(getattr(module, key) is None for key in ("running_mean", "running_var", "num_batches_tracked"))
                if module.track_running_stats != running or (running and not buffers_present) or (not running and not buffers_absent):
                    raise RuntimeError(f"BN policy mismatch at {name}, expected {self.bn_policy}")
                if not self.training and module.training:
                    raise RuntimeError(f"BN module {name} is training during model evaluation")

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        pretrained = list(self.backbone.parameters())
        if self.track == "N":
            pretrained += list(self.decode_head.parameters())
            new = list(self.referral.parameters())
        else:
            new = list(self.classifier.parameters())
        return [parameter for parameter in pretrained if parameter.requires_grad], [parameter for parameter in new if parameter.requires_grad]

    def forward(self, query: Tensor) -> dict[str, Tensor]:
        if query.ndim != 4 or query.shape[1] != 4 or query.shape[0] < 1:
            raise ValueError("D/N accept a nonempty batch of four-channel query tensors only")
        if query.dtype != torch.float32:
            raise ValueError("D/N require float32 query tensors")
        if self.track == "D" and not self.training and query.shape[0] != 1:
            raise ValueError("D batch-statistics evaluation requires batch size one")
        self._check_bn_policy()
        features = encode_query(self.backbone, query)
        if self.track in ("D", "D_running", "D_running_pilot"):
            pooled = features[-1].mean(dim=(2, 3))
            return {"attack_logit": self.classifier(pooled).squeeze(1)}
        projected = self.decode_head.base_forward(features)
        absent = torch.zeros_like(projected)
        fused = self.decode_head.discriminator(self.decode_head.neck_layer(projected, absent, "concat"))
        map_logit = self.decode_head.cls_seg(fused)
        return {"attack_logit": self.referral(fused, map_logit), "map_logit": map_logit}


def build_anchor_changer(track: str, levir_path: Path, expected_sha256: str, head_seed: int) -> nn.Module:
    """Verify offline LEVIR tensors, reproduce historical initialization, retain active modules."""
    if track not in ("D", "D_running", "D_running_pilot", "N"):
        raise ValueError(f"Unknown Changer pilot track {track!r}, expected D, D_running, D_running_pilot or N")
    path = Path(levir_path).resolve()
    source, digest = _verified_checkpoint(path, expected_sha256)
    with _seeded_construction(head_seed):
        historical = changerex.ChangerEx(fusion="single_noex", in_channels=4)
        historical.init_opencd_weights()
        provenance = _load_declared_levir(historical, source)
        model = AnchorChanger(historical, track, head_seed)
        active_prefixes = ("backbone.", "decode_head.") if track == "N" else ("backbone.",)
        active_state = {name: tensor for name, tensor in model.state_dict().items() if name.startswith(active_prefixes)}
        provenance["pretrained_tensor_sha256"] = _tensor_sha256(active_state)
    provenance.update({
        "requested_initialization": "levir",
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_loading_policy": "official_hash_bound_pickle_v1" if digest == _TRUSTED_LEVIR_SHA256 else "restricted_weights_only_v1",
        "head_seed": head_seed,
        "tensor_hash_format": "sorted_name_shape_dtype_compact_json_newline_then_c_order_bytes_v1",
        "bn_policy": model.bn_policy,
        "normalization_audit": _normalization_audit(model),
        "active_pretrained_keys": sorted(active_state),
        "active_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "new_parameter_count": sum(parameter.numel() for parameter in model.parameter_groups()[1]),
    })
    model.initialization_provenance = provenance
    return model
