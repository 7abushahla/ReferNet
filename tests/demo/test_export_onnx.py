# tests/demo/test_export_onnx.py
import os, sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "demo")]
CKPT = Path(os.environ.get("DEMO_CKPT_DIR", ROOT / "demo/ckpt"))
pytestmark = pytest.mark.skipif(not (CKPT / "on/best.pt").exists(), reason="deployment checkpoints not present")


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 4, 512, 800, generator=g)
    q[:, 3] = (torch.rand(1, 512, 800, generator=g) > 0.7).float()
    r = torch.randn(1, 3, 512, 800, generator=g)
    ov = torch.zeros(1, 1, 512, 800); ov[..., 40:480, 30:770] = 1
    return q, r, ov


def test_on_wrapper_matches_model():
    from bench_cpu_deploy import load_arm
    from export_onnx import OnExport
    model, _ = load_arm(CKPT / "on/best.pt", CKPT / "weights.pth", "on")
    q, r, ov = inputs()
    with torch.no_grad():
        ref = model(q, overlap=ov, reference=r, context_available=torch.tensor([True]))
        ql, corr, local = OnExport(model)(q, r, ov)
    assert torch.allclose(ql, ref["query_logit"], atol=1e-5)
    assert torch.allclose(corr, ref["correction_logit"], atol=1e-5)
    assert torch.allclose(ql + corr, ref["attack_logit"], atol=1e-5)
    assert local.shape == (1, 32, 64, 100)


def test_off_wrapper_matches_model():
    from bench_cpu_deploy import load_arm
    from export_onnx import OffExport
    model, _ = load_arm(CKPT / "off/best.pt", CKPT / "weights.pth", "off")
    q, _, ov = inputs(1)
    with torch.no_grad():
        ref = model(q, overlap=ov, reference=None, context_available=torch.tensor([True]))
        ql = OffExport(model)(q)
    assert torch.allclose(ql, ref["attack_logit"], atol=1e-5)


def test_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    from bench_cpu_deploy import load_arm
    from export_onnx import OnExport, OffExport, export
    info = export(CKPT, tmp_path)
    assert len(info["head"]["w_mean"]) == 32 and len(info["head"]["w_max"]) == 32
    q, r, ov = inputs(2)
    on = ort.InferenceSession(str(tmp_path / "refernet_on.onnx"), providers=["CPUExecutionProvider"])
    ql, corr, local = on.run(None, {"query": q.numpy(), "reference": r.numpy(), "overlap": ov.numpy()})
    with torch.no_grad():
        tql, tcorr, tlocal = OnExport(load_arm(CKPT / "on/best.pt", CKPT / "weights.pth", "on")[0])(q, r, ov)
        toff = OffExport(load_arm(CKPT / "off/best.pt", CKPT / "weights.pth", "off")[0])(q)
    np.testing.assert_allclose(ql, tql.numpy(), atol=1e-4)
    np.testing.assert_allclose(corr, tcorr.numpy(), atol=1e-4)
    np.testing.assert_allclose(local, tlocal.numpy(), atol=1e-3)
    off = ort.InferenceSession(str(tmp_path / "refernet_off.onnx"), providers=["CPUExecutionProvider"])
    (oql,) = off.run(None, {"query": q.numpy()})
    np.testing.assert_allclose(oql, toff.numpy(), atol=1e-4)
