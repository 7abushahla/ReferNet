"""Shape, loss and gradient checks for models/ and losses.py: python tests/test_model.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from refer_or_clear.losses import referral_loss  # noqa: E402
from refer_or_clear.models import build_model, count_params  # noqa: E402

torch.manual_seed(0)


def _batch(b, h, w):
    x = torch.randn(b, 8, h, w)
    label = torch.tensor([0.0, 1.0])[:b]
    mask = (torch.rand(b, 1, h, w) > 0.9).float() * label.view(-1, 1, 1, 1)
    valid = torch.ones(b, 1, h, w)
    valid[:, :, :, -16:] = 0
    return x, label, mask, valid


def check_fusion(fusion, b=2, h=256, w=400):
    model = build_model(fusion=fusion, init="scratch")
    model.train()
    x, label, mask, valid = _batch(b, h, w)
    out = model(x)
    assert out["logit"].shape == (b,), out["logit"].shape
    assert out["map"].shape == (b, 1, h // 4, w // 4), out["map"].shape
    total, parts = referral_loss(out["logit"], out["map"], label, mask, valid)
    assert torch.isfinite(total), total
    assert set(parts) == {"bce_image", "bce_map", "dice_map"} and all(v >= 0 for v in parts.values()), parts
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "a parameter received no gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    pre, new = model.parameter_groups()
    assert len(pre) + len(new) == len(list(model.parameters()))
    print(f"{fusion:11s} params {count_params(model):>12,d}  loss {total.item():.4f}  parts {parts}")
    return model


def check_stem_widening():
    m = build_model(fusion="difference", init="scratch", in_channels=4)
    w = m.encoder.stem[0].weight
    assert w.shape == (64, 4, 7, 7) and torch.count_nonzero(w[:, 3]) == 0, "extra stem channel must be zero"
    m = build_model(fusion="stacked", init="scratch", in_channels=4)
    assert m.encoder.stem[0].weight.shape == (64, 8, 7, 7)


def check_single_ignores_reference():
    m = build_model(fusion="single", init="scratch").eval()
    x, *_ = _batch(2, 128, 160)
    y = x.clone()
    y[:, 4:] = torch.randn_like(y[:, 4:])
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a["logit"], b["logit"]) and torch.allclose(a["map"], b["map"])


def check_changerex_raises():
    try:
        build_model(arch="changerex_r18")
    except NotImplementedError as e:
        assert "Open-CD adapter" in str(e)
    else:
        raise AssertionError("changerex_r18 must raise NotImplementedError")


def check_imagenet_init():
    try:
        m = build_model(fusion="difference", init="imagenet")
    except Exception as e:  # offline without a cache
        print(f"imagenet init skipped: {type(e).__name__}: {e}")
        return
    w = m.encoder.stem[0].weight
    assert w.shape == (64, 4, 7, 7) and torch.count_nonzero(w[:, 3]) == 0
    print("imagenet init ok (or fell back with a warning)")


def time_full_size(b=2, h=512, w=800):
    model = build_model(fusion="difference", init="scratch").train()
    x, label, mask, valid = _batch(b, h, w)
    t0 = time.perf_counter()
    out = model(x)
    total, _ = referral_loss(out["logit"], out["map"], label, mask, valid)
    total.backward()
    print(f"difference fwd+bwd at ({b}, 8, {h}, {w}) on CPU: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    for fusion in ("single", "stacked", "difference", "attention"):
        check_fusion(fusion)
    check_stem_widening()
    check_single_ignores_reference()
    check_changerex_raises()
    check_imagenet_init()
    time_full_size()
    print("all model tests passed")
