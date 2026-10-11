"""Fixed-checkpoint objective-gradient diagnostics and training-only selection."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def diagnostic():
    assert (ROOT / "scripts/diagnose_copy_gradients.py").exists(), "gradient diagnostic missing"
    import diagnose_copy_gradients
    return diagnose_copy_gradients


def test_gradient_geometry():
    module = diagnostic()
    weight = torch.tensor([1., 2.], requires_grad=True)
    unused = torch.tensor([3.], requires_grad=True)
    image_loss = weight.square().sum()
    localization_loss = -3 * image_loss
    first = torch.autograd.grad(image_loss, (weight, unused), retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(localization_loss, (weight, unused), allow_unused=True)
    result = module.gradient_geometry(first, second)
    assert abs(result["cosine"] + 1) < 1e-6
    assert abs(result["localization_to_image_norm"] - 3) < 1e-6
    assert weight.grad is None and unused.grad is None
    zero = module.gradient_geometry((torch.zeros(2),), (torch.zeros(2),))
    assert zero["cosine"] is None and zero["localization_to_image_norm"] is None


def test_train_only_design_matching():
    module = diagnostic()
    pairs = []
    for design in ("first", "second"):
        for family, count in (("text_copy", 2), ("text_inpaint", 4), ("", 4)):
            for index in range(count):
                pairs.append(dict(image_id=f"{design}-{family}-{index}", design_id=design,
                                  attack_family=family, label=int(bool(family)), wrong_design=False))
    selected = module.select_training_pairs(pairs, role="train", seed=42)
    assert selected == module.select_training_pairs(list(reversed(pairs)), role="train", seed=42)
    assert len(selected) == 12
    from collections import Counter
    for family in ("text_copy", "text_inpaint", ""):
        assert Counter(row["design_id"] for row in selected if row["attack_family"] == family) == {"first": 2, "second": 2}
    for role in ("val", "test"):
        try:
            module.select_training_pairs(pairs, role=role, seed=42)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted non-training examples")


if __name__ == "__main__":
    test_gradient_geometry()
    test_train_only_design_matching()
    print("ok: 2 copy-gradient tests")
