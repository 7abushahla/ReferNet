"""Pure query classifier with identical local-head tensors in both arms.

Reference-off has inactive input slices; reference-on adds encoder computation.
This matches local-head tensor capacity, not exact functional capacity.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from .anchor_changer import (
    AnchorChanger,
    _seeded_construction,
    _tensor_sha256,
    build_anchor_changer,
    encode_query,
)


class _LocalAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(128, 32, 1)
        self.local_head = nn.Sequential(
            nn.Conv2d(98, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.correction_head = nn.Linear(64, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    @staticmethod
    def _masked_pool(features: Tensor, mask: Tensor) -> Tensor:
        valid = mask.bool()
        count = valid.sum(dim=(2, 3)).clamp_min(1)
        mean = features.masked_fill(~valid, 0).sum(dim=(2, 3)) / count
        maximum = features.masked_fill(~valid, -torch.inf).amax(dim=(2, 3))
        maximum = torch.where(valid.flatten(1).any(dim=1, keepdim=True), maximum, 0)
        return torch.cat((mean, maximum), dim=1)

    def forward(
        self,
        query: Tensor,
        reference: Tensor,
        difference: Tensor,
        zone: Tensor,
        mask: Tensor,
    ) -> Tensor:
        comparison = torch.cat(
            (
                query.masked_fill(~mask, 0),
                reference.masked_fill(~mask, 0),
                difference.masked_fill(~mask, 0),
                zone.masked_fill(~mask, 0),
                mask.to(query.dtype),
            ),
            dim=1,
        )
        local = self.local_head(comparison)
        return self.correction_head(self._masked_pool(local, mask)).squeeze(1)


class DirectReferenceClassifier(nn.Module):
    architecture_id = "anchor_direct_reference_ia18_v1"
    normalization_policy = "shared_sequential_train_bn_stored_eval_v1"

    reference_bn_policies = ("shared", "stored")

    def __init__(
        self,
        base: AnchorChanger,
        *,
        reference_enabled: bool,
        adapter_seed: int = 42,
        local_enabled: bool = True,
        reference_bn: str = "shared",
    ):
        """Global query classifier with an optional local reference correction.

        ``local_enabled=False`` keeps the adapter tensors (same seeded
        initialization, so both arms share identical initial state) but
        freezes them and bypasses the local branch: the output is the global
        query logit alone. ``reference_bn="stored"`` runs the reference prefix
        under the stored BatchNorm statistics without updating them, so only
        query images shape the running moments used at evaluation.
        """
        super().__init__()
        if not isinstance(base, AnchorChanger) or base.track != "D_running_pilot" or base.bn_policy != "running":
            raise ValueError("Direct reference requires an internal D_running_pilot AnchorChanger")
        if type(reference_enabled) is not bool:
            raise ValueError("reference_enabled must be bool")
        if type(adapter_seed) is not int or not 0 <= adapter_seed < 2**32:
            raise ValueError("adapter_seed must be an integer in [0, 2**32)")
        if type(local_enabled) is not bool:
            raise ValueError("local_enabled must be bool")
        if reference_bn not in self.reference_bn_policies:
            raise ValueError("reference_bn must be 'shared' or 'stored'")
        if reference_enabled and not local_enabled:
            raise ValueError("a reference-enabled arm requires the local branch")
        self.base = base
        self.reference_enabled = reference_enabled
        self.local_enabled = local_enabled
        self.reference_bn = reference_bn
        with _seeded_construction(adapter_seed):
            self.adapter = _LocalAdapter()
        if not local_enabled:
            for parameter in self.adapter.parameters():
                parameter.requires_grad_(False)
        self.initialization_provenance = {
            "base": deepcopy(base.initialization_provenance),
            "architecture_id": self.architecture_id,
            "base_architecture_id": base.architecture_id,
            "normalization_policy": self.normalization_policy,
            "reference_enabled": reference_enabled,
            "local_enabled": local_enabled,
            "reference_bn": reference_bn,
            "adapter_seed": adapter_seed,
            "adapter_tensor_sha256": _tensor_sha256(self.adapter.state_dict()),
        }

    def _reference_prefix(self, reference_input: Tensor) -> Tensor:
        backbone = self.base.backbone
        prefix = (backbone.stem, backbone.layer1, backbone.layer2)
        normalizers = [module for stage in prefix for module in stage.modules()
                       if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        restore = [module.training for module in normalizers]
        if self.reference_bn == "stored":
            for module in normalizers:
                module.eval()
        try:
            return backbone.layer2(backbone.layer1(backbone.maxpool(backbone.stem(reference_input))))
        finally:
            for module, training in zip(normalizers, restore):
                module.train(training)

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        pretrained = [parameter for parameter in self.base.backbone.parameters() if parameter.requires_grad]
        new = [
            parameter
            for module in (self.base.classifier, self.adapter)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return pretrained, new

    def _validate_inputs(
        self,
        query: Tensor,
        overlap: Tensor,
        reference: Tensor | None,
        context_available: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(query, Tensor) or query.ndim != 4 or query.shape[1] != 4:
            raise ValueError("query must have shape B4HW")
        batch, _, height, width = query.shape
        parameter = next(self.base.parameters())
        if query.dtype != torch.float32 or query.device != parameter.device or parameter.dtype != torch.float32:
            raise ValueError("query and model must share a device and use float32")
        if batch < 1 or height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError("query must have a nonempty batch and spatial dimensions divisible by 32")
        if not torch.isfinite(query).all() or not ((query[:, 3:4] >= 0) & (query[:, 3:4] <= 1)).all():
            raise ValueError("query must be finite with its zone in [0, 1]")
        if not isinstance(overlap, Tensor) or overlap.shape != (batch, 1, height, width) or overlap.device != query.device or overlap.dtype != torch.float32:
            raise ValueError("overlap must have shape B1HW and float32 dtype on the query device")
        if not ((overlap == 0) | (overlap == 1)).all():
            raise ValueError("overlap must be binary")
        if not isinstance(context_available, Tensor) or context_available.shape != (batch,) or context_available.dtype != torch.bool or context_available.device != query.device:
            raise ValueError("context_available must have shape B and bool dtype on the query device")
        mask = functional.interpolate(overlap, size=(height // 8, width // 8), mode="nearest").bool()
        available = context_available & mask.flatten(1).any(dim=1)
        if self.reference_enabled and available.any():
            if not isinstance(reference, Tensor) or reference.shape != (batch, 3, height, width):
                raise ValueError("available reference must have shape B3HW")
            if reference.device != query.device or reference.dtype != torch.float32:
                raise ValueError("available reference must use float32 on the query device")
            if not torch.isfinite(reference[available]).all():
                raise ValueError("available reference rows must be finite")
        return mask & available[:, None, None, None], available

    def forward(
        self,
        query: Tensor,
        *,
        overlap: Tensor,
        reference: Tensor | None,
        context_available: Tensor,
    ) -> dict[str, Tensor]:
        mask, available = self._validate_inputs(query, overlap, reference, context_available)
        self.base._check_bn_policy()
        stages = encode_query(self.base.backbone, query)
        query_logit = self.base.classifier(stages[-1].mean(dim=(2, 3))).squeeze(1)
        correction_logit = torch.zeros_like(query_logit)
        if self.local_enabled and available.any():
            indices = available.nonzero(as_tuple=True)[0]
            projected_query = self.adapter.projection(stages[1][available])
            projected_reference = torch.zeros_like(projected_query)
            difference = torch.zeros_like(projected_query)
            if self.reference_enabled:
                reference_input = torch.cat((reference[available], query[available, 3:4]), dim=1)
                reference_features = self._reference_prefix(reference_input)
                projected_reference = self.adapter.projection(reference_features)
                difference = (projected_query - projected_reference).abs()
            zone = functional.interpolate(query[available, 3:4].clamp(0, 1), size=projected_query.shape[2:], mode="nearest")
            correction = self.adapter(projected_query, projected_reference, difference, zone, mask[available])
            correction_logit = correction_logit.index_copy(0, indices, correction)
        return {
            "attack_logit": query_logit + correction_logit,
            "query_logit": query_logit,
            "correction_logit": correction_logit,
        }


def build_direct_reference_model(
    levir_path: Path,
    expected_sha256: str,
    *,
    reference_enabled: bool,
    head_seed: int = 42,
    adapter_seed: int = 42,
    local_enabled: bool = True,
    reference_bn: str = "shared",
) -> DirectReferenceClassifier:
    base = build_anchor_changer("D_running_pilot", levir_path, expected_sha256, head_seed)
    return DirectReferenceClassifier(base, reference_enabled=reference_enabled, adapter_seed=adapter_seed,
                                     local_enabled=local_enabled, reference_bn=reference_bn)
