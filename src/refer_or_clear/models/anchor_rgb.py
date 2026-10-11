import hashlib
import io
import json
from pathlib import Path

import torch
from torch import Tensor, nn
import torchvision.models

from refer_or_clear.anchor_contract import contract_for


_IMAGENET1K_V1_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"


def _tensor_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode("utf-8")
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class AnchorRGB(nn.Module):
    def __init__(self, backbone: nn.Module, initialization_provenance: dict):
        super().__init__()
        self.backbone = backbone
        self.architecture_id = contract_for("R").architecture_id
        self.track = "R"
        self.initialization_provenance = initialization_provenance

    def forward(self, query: Tensor) -> dict[str, Tensor]:
        class_logits = self.backbone(query)
        return {
            "class_logits": class_logits,
            "attack_logit": class_logits[:, 1] - class_logits[:, 0],
        }

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        head = [parameter for parameter in self.backbone.fc.parameters() if parameter.requires_grad]
        head_ids = {id(parameter) for parameter in head}
        pretrained = [
            parameter for parameter in self.backbone.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        return pretrained, head


def build_anchor_rgb(weights_path: Path, expected_sha256: str, head_seed: int) -> nn.Module:
    """Load staged torchvision IMAGENET1K_V1 tensors without network access.

    Only the full official source SHA256 permits absent historical BN counters
    to become scalar int64 zeros. Modern BN metadata never permits migration.
    Loaded keys and their tensor hash describe the original checkpoint tensors.
    Source missing keys record original omissions. Missing and unexpected keys
    report only unresolved differences after strict normalized loading.
    The normalized tensor hash includes declared synthesized counters and the
    original 1,000-class head, before its replacement.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Requested ImageNet checkpoint is unavailable: {weights_path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        raise ValueError("Expected SHA256 must be a full 64-character hexadecimal digest")
    checkpoint_bytes = weights_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha256 != expected_sha256.lower():
        raise ValueError(f"Checkpoint SHA256 mismatch for {weights_path}: {checkpoint_sha256}")
    try:
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Cannot deserialize requested ImageNet checkpoint {weights_path}") from error
    if not isinstance(state, dict) or any(not isinstance(name, str) for name in state):
        raise ValueError("ImageNet checkpoint must contain a plain tensor state dict")
    with torch.random.fork_rng(devices=[]):
        backbone = torchvision.models.resnet50(weights=None)
        expected = backbone.state_dict()
        expected_keys = set(expected)
        supplied_keys = set(state)
        missing_keys = sorted(expected_keys - supplied_keys)
        unexpected_keys = sorted(supplied_keys - expected_keys)
        synthesized = {}
        if missing_keys and checkpoint_sha256 == _IMAGENET1K_V1_SHA256:
            metadata = getattr(state, "_metadata", {})
            historical_counter_keys = {
                f"{name}.num_batches_tracked"
                for name, module in backbone.named_modules()
                if isinstance(module, nn.BatchNorm2d) and module.track_running_stats
                and metadata.get(name, {}).get("version") in (None, 0, 1)
            }
            synthesized = {
                name: torch.zeros_like(expected[name])
                for name in missing_keys if name in historical_counter_keys
            }
        if set(missing_keys) != set(synthesized) or unexpected_keys:
            raise ValueError(f"Checkpoint keys mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
        normalized_state = {**state, **synthesized}
        for name in sorted(expected_keys):
            tensor = normalized_state[name]
            if not isinstance(tensor, Tensor):
                raise ValueError(f"Checkpoint tensor {name} is not a Tensor")
            if tensor.shape != expected[name].shape:
                raise ValueError(f"Checkpoint tensor {name} shape mismatch: {tuple(tensor.shape)} != {tuple(expected[name].shape)}")
            if tensor.dtype != expected[name].dtype:
                raise ValueError(f"Checkpoint tensor {name} dtype mismatch: {tensor.dtype} != {expected[name].dtype}")
        load_result = backbone.load_state_dict(normalized_state, strict=True)
        loaded_state = backbone.state_dict()
        pretrained_state = {name: tensor for name, tensor in loaded_state.items() if name not in {"fc.weight", "fc.bias"}}
        provenance = {
            "requested_initialization": "IMAGENET1K_V1",
            "checkpoint_path": str(weights_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "loaded_keys": sorted(supplied_keys),
            "source_missing_keys": missing_keys,
            "missing_keys": sorted(load_result.missing_keys),
            "unexpected_keys": sorted(load_result.unexpected_keys),
            "synthesized_keys": sorted(synthesized),
            "synthesized_values": {name: tensor.item() for name, tensor in synthesized.items()},
            "migration_policy": "official_imagenet1k_v1_missing_bn_counters_zero_v1" if synthesized else None,
            "pretrained_tensor_sha256": _tensor_sha256(pretrained_state),
            "pretrained_keys": sorted(pretrained_state),
            "loaded_tensor_sha256": _tensor_sha256(state),
            "normalized_tensor_sha256": _tensor_sha256(loaded_state),
            "tensor_hash_format": "sorted_name_shape_dtype_json_length_u64be_then_c_order_bytes_length_u64be_v1",
            "head_seed": head_seed,
            "source_url": torchvision.models.ResNet50_Weights.IMAGENET1K_V1.url,
        }
        torch.random.default_generator.manual_seed(head_seed)
        backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    return AnchorRGB(backbone, provenance)
