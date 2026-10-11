"""One-image inference policy for the shared ChangerEx encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.models.changerex import set_encoder_bn_policy  # noqa: E402


class TwoBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.BatchNorm2d(1)
        self.decode_head = nn.BatchNorm2d(1)

    def forward(self, query, reference):
        return self.decode_head(self.backbone(query) - self.backbone(reference))


def test_encoder_batch_stats_preserves_parameters_and_requires_batch_one():
    model = TwoBranchModel().eval()
    model.backbone.running_mean.fill_(4)
    before = {name: value.clone() for name, value in model.named_parameters()}
    set_encoder_bn_policy(model, "batch_stats")
    assert model.backbone.running_mean is None
    assert model.backbone.running_var is None
    assert model.backbone.num_batches_tracked is None
    assert not model.backbone.track_running_stats
    assert model.decode_head.track_running_stats
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters())
    query = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    reference = torch.ones_like(query)
    other = torch.randn_like(query) * 20
    with torch.no_grad():
        alone = model(query, reference)
        together = model(torch.cat([query, other]), torch.cat([reference, other]))[:1]
    assert not torch.allclose(alone, together, atol=1e-5)
    state = model.state_dict()
    assert "backbone.running_mean" not in state
    assert "decode_head.running_mean" in state
    restored = TwoBranchModel()
    set_encoder_bn_policy(restored, "batch_stats")
    restored.load_state_dict(state, strict=True)


def test_running_policy_is_unchanged_and_invalid_policy_fails():
    model = TwoBranchModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    set_encoder_bn_policy(model, "running")
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    try:
        set_encoder_bn_policy(model, "unsupported")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid BN policy was accepted")


if __name__ == "__main__":
    test_encoder_batch_stats_preserves_parameters_and_requires_batch_one()
    test_running_policy_is_unchanged_and_invalid_policy_fails()
    print("ok: 2 BN policy tests")
