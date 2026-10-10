"""ChangerEx attention variant: shapes, loss, gradients, parameter counts, step-0 equality with the
stock variant, and a timing line. Needs the Open-CD stack (docs/implementation-plan.md, Section 9).
Run: PYTHONPATH=src python tests/test_changerex_attention.py  (forward/backward on cuda when
available, the equality check always on CPU in float32)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from refer_or_clear.losses import referral_loss  # noqa: E402
from refer_or_clear.models.changerex import build_changerex  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _batch(b, h, w, device="cpu"):
    x = torch.randn(b, 8, h, w)
    label = (torch.arange(b) % 2).float()
    mask = (torch.rand(b, 1, h, w) > 0.9).float() * label.view(-1, 1, 1, 1)
    valid = torch.ones(b, 1, h, w)
    valid[:, :, :, -16:] = 0
    return [t.to(device) for t in (x, label, mask, valid)]


def _count(module):
    return sum(p.numel() for p in module.parameters())


def check_fusion(fusion, b=2, h=256, w=400):
    torch.manual_seed(0)
    model = build_changerex(fusion=fusion, init="scratch").to(DEVICE).train()
    x, label, mask, valid = _batch(b, h, w, DEVICE)
    out = model(x)
    assert out["logit"].shape == (b,), out["logit"].shape
    assert out["map"].shape == (b, 1, h // 4, w // 4), out["map"].shape
    total, parts = referral_loss(out["logit"], out["map"], label, mask, valid)
    assert torch.isfinite(total), total
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "a parameter received no gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    pre, new = model.parameter_groups()
    assert len(pre) + len(new) == len(list(model.parameters()))
    n_attn = _count(model.attention) if model.attention is not None else 0
    print(f"{fusion:10s} params total {_count(model):>11,d}  backbone {_count(model.backbone):,d}  "
          f"head {_count(model.decode_head):,d}  referral {_count(model.referral):,d}  attention {n_attn:,d}  "
          f"new-group tensors {len(new)}  loss {total.item():.4f}  parts {parts}")
    if model.attention is not None:
        for name, m in model.attention.items():
            print(f"             {name:8s} {_count(m):,d}")
    return model


def check_step0_equality(b=2, h=256, w=400):
    """Zero-initialised output projections: the attention model computes the stock model at step 0."""
    torch.manual_seed(0)
    ref = build_changerex(fusion="difference", init="scratch").eval()
    torch.manual_seed(0)
    att = build_changerex(fusion="attention", init="scratch").eval()
    sd_ref, sd_att = ref.state_dict(), att.state_dict()
    assert set(sd_ref) <= set(sd_att)
    assert all(torch.equal(sd_ref[k], sd_att[k]) for k in sd_ref), "shared tensors differ after the same seed"
    torch.manual_seed(1)
    x, *_ = _batch(b, h, w)
    with torch.no_grad():
        a, c = ref(x), att(x)
    d_map, d_logit = (a["map"] - c["map"]).abs().max().item(), (a["logit"] - c["logit"]).abs().max().item()
    assert torch.allclose(a["map"], c["map"], atol=1e-5) and torch.allclose(a["logit"], c["logit"], atol=1e-5)
    print(f"step-0 equality (CPU, eval): max |map diff| {d_map:.2e}, max |logit diff| {d_logit:.2e}, "
          f"{len(sd_ref)} shared state tensors bit-identical")
    # The attention modules must be live: perturb one output projection and the outputs must move.
    with torch.no_grad():
        att.attention["stage32"].cross_attn.to_out.weight.fill_(1e-2)
        e = att(x)
    assert not torch.allclose(a["map"], e["map"]), "attention path is not connected to the map"


def time_full_size(fusion, b=2, h=512, w=800, amp=False):
    """One timed fwd+bwd (after one warmup pass on cuda); AMP = bfloat16-free float16 autocast with a
    GradScaler-less backward, enough for a peak-memory figure. No optimizer step."""
    torch.manual_seed(0)
    model = build_changerex(fusion=fusion, init="scratch").to(DEVICE).train()
    x, label, mask, valid = _batch(b, h, w, DEVICE)

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and DEVICE == "cuda"):
            out = model(x)
            total, _ = referral_loss(out["logit"].float(), out["map"].float(), label, mask, valid)
        total.backward()
        return out

    if DEVICE == "cuda":
        step(), model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(), torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    mem = f", peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if DEVICE == "cuda" else ""
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"{fusion:10s} fwd+bwd at ({b}, 8, {h}, {w}) on {DEVICE}{', amp' if amp else ''}: "
          f"{time.perf_counter() - t0:.2f} s, map {tuple(out['map'].shape)}{mem}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, device {DEVICE}")
    for fusion in ("difference", "attention", "single"):
        check_fusion(fusion)
    check_step0_equality()
    for fusion in ("difference", "attention"):
        time_full_size(fusion)
    if DEVICE == "cuda":
        for fusion in ("difference", "attention"):
            time_full_size(fusion, b=8, amp=True)
    print("all changerex attention tests passed")
