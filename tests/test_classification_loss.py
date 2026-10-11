"""Classification-only training ignores localization targets and preserves image gradients."""
from pathlib import Path
import sys

import torch
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.losses import referral_loss


def test_zero_weight_does_not_read_dense_targets():
    logits = torch.tensor([-1., 2.], requires_grad=True)
    labels = torch.tensor([0., 1.])
    total, parts = referral_loss(logits, None, labels, None, None, mask_weight=0.)
    expected = functional.binary_cross_entropy_with_logits(logits, labels)
    assert torch.equal(total, expected)
    assert parts["bce_map"] == parts["dice_map"] == 0.
    actual_gradient = torch.autograd.grad(total, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_map_can_learn_through_classification_without_mask_supervision():
    feature = torch.tensor([[[[0., 2.], [1., -1.]]]], requires_grad=True)
    logit = feature.amax((1, 2, 3))
    total, _ = referral_loss(logit, feature, torch.tensor([0.]), None, None, mask_weight=0.)
    total.backward()
    assert feature.grad[0, 0, 0, 1] > 0
    assert torch.count_nonzero(feature.grad) == 1


if __name__ == "__main__":
    test_zero_weight_does_not_read_dense_targets()
    test_map_can_learn_through_classification_without_mask_supervision()
    print("ok: classification-only loss and map gradients")
