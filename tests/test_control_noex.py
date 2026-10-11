"""The fair control: exchange layers off, reference blank: python tests/test_control_noex.py
(builds ChangerEx, so it needs the Open-CD stack: run on the cluster)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.models.changerex import build_changerex  # noqa: E402


def test_no_exchange_and_same_parameters():
    torch.manual_seed(0)
    ctl = build_changerex(fusion="single_noex", init="scratch").eval()
    torch.manual_seed(0)
    ref = build_changerex(fusion="difference", init="scratch").eval()
    n_ctl, n_ref = (sum(p.numel() for p in m.parameters()) for m in (ctl, ref))
    assert n_ctl == n_ref, (n_ctl, n_ref)
    names = [type(m).__name__ for m in ctl.backbone.ccs] if hasattr(ctl.backbone, "ccs") else []
    assert not any("Exchange" in n for n in names), names
    assert any("Exchange" in type(m).__name__ for m in ref.backbone.ccs), "stock model lost its exchange"
    x = torch.randn(2, 8, 256, 400)
    q, r = ctl.split(x)
    assert torch.equal(r[:, :3], torch.zeros_like(r[:, :3])) and torch.equal(r[:, 3], x[:, 3])
    out = ctl(x)
    assert out["logit"].shape == (2, 1) or out["logit"].numel() == 2, out["logit"].shape
    # the control must not depend on the reference planes at all
    x2 = x.clone(); x2[:, 4:] = torch.randn_like(x2[:, 4:])
    assert torch.allclose(ctl(x)["logit"], ctl(x2)["logit"]), "control reads the reference"


if __name__ == "__main__":
    test_no_exchange_and_same_parameters()
    print("ok")
