# ReferNet Demo Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static GitHub Pages site in `site/` where a visitor picks a design reference and a document, chooses with or without the reference, and runs the paper's fold-0 deployment model in the browser to get a percentage, Refer or Clear, the inference time, and (with the reference) a difference map.

**Architecture:** Offline Python tools in `demo/` export the two deployment checkpoints to ONNX and dump each sample's exact network inputs (registered query, warped reference, zone, overlap) from the paper's own evaluation datasets, verified against the paper's locked logits. The browser decodes those inputs from lossless PNGs, assembles tensors, and runs ONNX Runtime Web. Pure browser logic lives in small ES modules tested with `node --test`.

**Tech Stack:** Python 3.10, PyTorch 2.1, onnx 1.16.1, onnxruntime 1.18.1 (CPU, tests only), OpenCV; plain HTML/CSS/ES modules; onnxruntime-web 1.20.1, UPNG.js 2.1.0 and pako 1.0.11 from jsdelivr; GitHub Actions Pages deploy.

**Spec:** `docs/superpowers/specs/2026-10-15-refernet-demo-design.md`

## Global Constraints

- Model: fold-0 deployment pair of Protocol A (`ckpt/off/best.pt`, `ckpt/on/best.pt`, encoder weights `ckpt/weights.pth`, SHA256 `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`), FP32, two ONNX files `site/model/refernet_on.onnx` and `site/model/refernet_off.onnx`.
- Inputs: 512×800 canvas; query = 4 channels `[ImageNet-normalized RGB; zone]`; reference = 3 channels ImageNet-normalized RGB; overlap = 1 channel binary. ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Decision: p = σ(ℓ_q + Δℓ) with the reference, p = σ(ℓ_q) without; Refer when p > 0.5, Clear otherwise (ties Clear, as in the paper's "bonafide_score >= 0.5 is genuine").
- Samples: compression-equalized (JPEG quality 75, the paper's q75 evaluation), only from the paper's locked fold-0 predictions (SIDTD Protocol A fold-0 test partition; IDNet Protocol C queries), only rows with reference context; one genuine document and one document per text family for each design (none only where the paper's population has none: SIDTD Finland has no copy-and-replace), a proof-of-concept library; families SIDTD `genuine, text_inpaint, text_copy`, IDNet adds `text_replace`; selection seed `20261015`. Expected total 61 (SIDTD 29, IDNet 32).
- Display names: families "Genuine", "Inpaint-and-rewrite", "Copy-and-replace", "In-place text replacement". The word "enrollment" never appears in visible text or the manifest.
- UI copy: no logits or loss terms are shown; the result is "NN% likelihood of alteration", "REFER"/"CLEAR", "Inference time NNN ms", ground truth with ✓/✗.
- Difference map note text (verbatim): "The map in step 4 is not part of the paper. It is shown for demonstration only. It is computed in your browser from the same run, from the comparison features weighted by the correction layer, to indicate which region drove the reference's correction, and it has not been evaluated as a localization method."
- Licenses in the footer: SIDTD CC BY 4.0, MIDV-2020 images CC BY-SA 2.5, IDNet CC0.
- No build step for the site; CDN script versions pinned exactly as above.
- Nothing is pushed and Pages is not enabled until the separate rename/publish task is done and the user says so.

## Review Focus

- Browser color management or canvas fingerprint noise altering decoded pixels, so scores silently drift from the paper. Expected: PNGs decode with UPNG (no canvas), and `verify.html` fails loudly on any mismatch. Pinned by Task 6 (`decodePng` path) and Task 8 (parity page).
- A document from another design selected, with either toggle state. Expected: the network does not run; REFER with "no reference context, sent to the second line". Pinned by Task 6 `decideMismatch` test and Task 7 manual step 4.
- No WebGPU (Safari, Firefox). Expected: WebAssembly fallback and the backend actually used is named. Pinned by Task 7 `pickProviders` test and manual step 1.
- First visit on a slow connection. Expected: Run disabled with a progress bar until the ON model (≈45 MB) is loaded and warmed up; OFF loads on first "Without reference". Pinned by Task 7 manual steps 3 and 6.
- Rapid repeated clicks on Run. Expected: one run at a time, button disabled during a run, the reported time excludes the warm-up. Pinned by Task 7 `RunGate` test and manual step 5.

---

## File Structure

```
demo/
  export_onnx.py         # OnExport/OffExport wrappers, ONNX export, head weights -> site/model/
  select_samples.py      # deterministic sample choice from the paper's locked predictions
  dump_inputs.py         # (cluster) exact network inputs from the paper's eval datasets, verified against the paper
  build_site_data.py     # PNGs + manifest.json into site/
  sources.json           # (Task 3) cluster paths of the fold-0 A and C evaluation requests
tests/demo/
  test_export_onnx.py  test_select_samples.py  test_build_site_data.py
site/
  index.html  style.css  verify.html
  js/tensors.js  js/decision.js  js/heatmap.js  js/library.js  js/runner.js  js/main.js  js/verify.js
  tests/*.test.mjs
  model/refernet_on.onnx  model/refernet_off.onnx  model/head.json
  designs/<design_id>/reference.png  designs/<design_id>/zone.png
  samples/<sample_id>/query.png  samples/<sample_id>/reference.png  samples/<sample_id>/masks.png
  manifest.json
.github/workflows/pages.yml
```

`<design_id>`: SIDTD `sidtd_<code>` (e.g. `sidtd_aze_passport`), IDNet `idnet_<CODE>` (e.g. `idnet_AZ`). `<sample_id>`: `<design_id>__<family>__<n>`, n = 1 (the suffix keeps room for more samples later).

---

### Task 1: ONNX export of the deployment pair

**Files:**
- Create: `demo/export_onnx.py`
- Test: `tests/demo/test_export_onnx.py`
- Modify: `.gitignore` (add `demo/ckpt/`)

**Interfaces:**
- Consumes: `scripts/bench_cpu_deploy.py::load_arm(path, weights, arm) -> (model, info)`; `refer_or_clear.models.anchor_changer.encode_query(backbone, query) -> list[Tensor]`.
- Produces: `OnExport(model)(query[1,4,512,800], reference[1,3,512,800], overlap[1,1,512,800]) -> (query_logit[1], correction[1], local[1,32,64,100])`; `OffExport(model)(query) -> query_logit[1]`; `export(ckpt_dir: Path, out_dir: Path) -> dict` writing `refernet_on.onnx` (inputs `query, reference, overlap`; outputs `query_logit, correction, local`), `refernet_off.onnx` (input `query`; output `query_logit`), `head.json` = `{"w_mean": [32 floats], "w_max": [32 floats], "bias": float}`.

- [ ] **Step 1: Copy the checkpoints from the benchmark laptop and install the export tools**

```bash
cd ~/Documents/GitHub/refer-or-clear
mkdir -p demo/ckpt/off demo/ckpt/on
scp aric@192.168.1.202:roc-bench/ckpt/off/best.pt demo/ckpt/off/best.pt
scp aric@192.168.1.202:roc-bench/ckpt/on/best.pt  demo/ckpt/on/best.pt
scp aric@192.168.1.202:roc-bench/ckpt/weights.pth demo/ckpt/weights.pth
shasum -a 256 demo/ckpt/weights.pth   # must start da3f5693 and end 3618
echo "demo/ckpt/" >> .gitignore
pip install onnx==1.16.1 onnxruntime==1.18.1
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/demo/test_export_onnx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_onnx'`

- [ ] **Step 4: Write the implementation**

```python
# demo/export_onnx.py
"""Export the fold-0 deployment pair (ReferNet OFF and ON) to ONNX for the demo website.

The wrappers re-express DirectReferenceClassifier.forward for one document whose reference context is available
(the site never runs the network without it), without the data-dependent branches that ONNX tracing cannot keep.
ON also returns the comparison features before pooling, from which the site draws the difference map.

Usage: python demo/export_onnx.py [--ckpt demo/ckpt] [--out site/model]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from bench_cpu_deploy import load_arm  # noqa: E402
from refer_or_clear.models.anchor_changer import encode_query  # noqa: E402


class OnExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model.reference_enabled and model.local_enabled and model.fused_query_weight == 1.0
        self.m = model

    def forward(self, query, reference, overlap):
        m, a = self.m, self.m.adapter
        stages = encode_query(m.base.backbone, query)
        query_logit = m.base.classifier(stages[-1].mean(dim=(2, 3))).squeeze(1)
        mask = F.interpolate(overlap, size=(query.shape[2] // 8, query.shape[3] // 8), mode="nearest") > 0.5
        pq = a.projection(stages[1])
        pr = a.projection(m._reference_prefix(torch.cat((reference, query[:, 3:4]), dim=1)))
        diff = (pq - pr).abs()
        zone = F.interpolate(query[:, 3:4].clamp(0, 1), size=pq.shape[2:], mode="nearest")
        comparison = torch.cat((pq.masked_fill(~mask, 0), pr.masked_fill(~mask, 0), diff.masked_fill(~mask, 0),
                                zone.masked_fill(~mask, 0), mask.to(pq.dtype)), dim=1)
        local = a.local_head(comparison)
        correction = a.correction_head(a._masked_pool(local, mask)).squeeze(1)
        return query_logit, correction, local


class OffExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert not model.local_enabled
        self.m = model

    def forward(self, query):
        stages = encode_query(self.m.base.backbone, query)
        return self.m.base.classifier(stages[-1].mean(dim=(2, 3))).squeeze(1)


def export(ckpt_dir: Path, out_dir: Path) -> dict:
    ckpt_dir, out_dir = Path(ckpt_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "weights.pth"
    on, _ = load_arm(ckpt_dir / "on/best.pt", weights, "on")
    off, _ = load_arm(ckpt_dir / "off/best.pt", weights, "off")
    q, r, ov = torch.zeros(1, 4, 512, 800), torch.zeros(1, 3, 512, 800), torch.ones(1, 1, 512, 800)
    with torch.no_grad():
        torch.onnx.export(OnExport(on), (q, r, ov), str(out_dir / "refernet_on.onnx"), opset_version=17,
                          input_names=["query", "reference", "overlap"],
                          output_names=["query_logit", "correction", "local"], do_constant_folding=True)
        torch.onnx.export(OffExport(off), (q,), str(out_dir / "refernet_off.onnx"), opset_version=17,
                          input_names=["query"], output_names=["query_logit"], do_constant_folding=True)
    w = on.adapter.correction_head.weight.detach().flatten().tolist()
    head = {"w_mean": w[:32], "w_max": w[32:], "bias": float(on.adapter.correction_head.bias.item())}
    (out_dir / "head.json").write_text(json.dumps(head))
    return {"head": head, "files": sorted(p.name for p in out_dir.iterdir())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "demo/ckpt")); ap.add_argument("--out", default=str(ROOT / "site/model"))
    a = ap.parse_args()
    print(json.dumps(export(Path(a.ckpt), Path(a.out))["files"]))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/demo/test_export_onnx.py -v`
Expected: 3 passed.

- [ ] **Step 6: Export into the site and commit**

```bash
python demo/export_onnx.py
ls -la site/model   # refernet_on.onnx and refernet_off.onnx about 45 MB each, head.json
git add .gitignore demo/export_onnx.py tests/demo/test_export_onnx.py site/model
git commit -m "Demo: ONNX export of the fold-0 deployment pair with the comparison-map output"
```

---

### Task 2: Deterministic sample selection from the locked predictions

**Files:**
- Create: `demo/select_samples.py`
- Test: `tests/demo/test_select_samples.py`

**Interfaces:**
- Produces: `select_samples(seed: int = 20261015, per_family: int = 1) -> list[dict]`, each dict `{"sample_id", "image_id", "design_id", "design_key", "dataset", "family", "label", "paper": {"on": {"query_logit", "correction_logit", "attack_logit"}, "off": {"attack_logit"}}}`; `eligible_counts() -> dict[(design_id, family), int]`; `select_mismatch(samples) -> list[dict]` (the `__1` genuine sample of `sidtd_esp_id`, `sidtd_srb_passport`, `idnet_AZ`, each with `"mismatch_demo": True`); constants `DESIGNS` (design_id → `{"key", "dataset", "name", "doc"}`), `FAMILIES` (family → display name), `FAMILIES_OF` (dataset → family list).

- [ ] **Step 1: Write the failing test**

```python
# tests/demo/test_select_samples.py
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from select_samples import DESIGNS, FAMILIES_OF, eligible_counts, select_mismatch, select_samples  # noqa: E402


def test_counts_follow_the_paper_population_and_are_deterministic():
    a, b = select_samples(), select_samples()
    assert a == b
    assert len(DESIGNS) == 18
    per, avail = Counter((s["design_id"], s["family"]) for s in a), eligible_counts()
    for d, meta in DESIGNS.items():
        for f in FAMILIES_OF[meta["dataset"]]:
            assert per[(d, f)] == min(1, avail.get((d, f), 0)), (d, f)
    assert per[("sidtd_fin_id", "text_copy")] == 0
    assert len(a) == 61
    assert len({s["image_id"] for s in a}) == len(a)


def test_labels_ids_and_paper_logits():
    for s in select_samples():
        assert s["label"] == (0 if s["family"] == "genuine" else 1)
        on = s["paper"]["on"]
        assert abs(on["query_logit"] + on["correction_logit"] - on["attack_logit"]) < 1e-3
        assert s["sample_id"].startswith(f"{s['design_id']}__{s['family']}__")


def test_mismatch_samples():
    m = select_mismatch(select_samples())
    assert [s["design_id"] for s in m] == ["sidtd_esp_id", "sidtd_srb_passport", "idnet_AZ"]
    assert all(s["family"] == "genuine" and s["mismatch_demo"] for s in m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/demo/test_select_samples.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'select_samples'`

- [ ] **Step 3: Write the implementation**

```python
# demo/select_samples.py
"""Pick the demo samples, by seed, from the paper's locked fold-0 compression-equalized (q75) predictions.

SIDTD: Protocol A fold-0 test partition (never trained on). IDNet: the Protocol C queries of the deployment model
(outside each design's reference set). Only rows with reference context are eligible. One per family and design (a proof-of-concept library);
none only where the paper's population has none. Wrong decisions are kept.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A_DIR = ROOT / "results/diagnostics/locked-predictions/protocol-a/fold0-v2noaux-fdb7351/evaluation-q75"
C_DIR = ROOT / "results/diagnostics/protocol-c-idnet"
FAMILIES = {"genuine": "Genuine", "text_inpaint": "Inpaint-and-rewrite", "text_copy": "Copy-and-replace",
            "text_replace": "In-place text replacement"}
FAMILIES_OF = {"sidtd": ["genuine", "text_inpaint", "text_copy"],
               "idnet": ["genuine", "text_inpaint", "text_copy", "text_replace"]}
_SIDTD = [("alb_id", "Albania", "identity card"), ("aze_passport", "Azerbaijan", "passport"),
          ("esp_id", "Spain", "identity card"), ("est_id", "Estonia", "identity card"),
          ("fin_id", "Finland", "identity card"), ("grc_passport", "Greece", "passport"),
          ("lva_passport", "Latvia", "passport"), ("rus_internalpassport", "Russia", "passport"),
          ("srb_passport", "Serbia", "passport"), ("svk_id", "Slovakia", "identity card")]
_IDNET = [("FIN", "Finland", "identity card"), ("GRC", "Greece", "passport"), ("ESP", "Spain", "identity card"),
          ("ALB", "Albania", "identity card"), ("AZ", "Arizona", "license"), ("SD", "South Dakota", "license"),
          ("WV", "West Virginia", "license"), ("NV", "Nevada", "license")]
DESIGNS = {**{f"sidtd_{k}": dict(key=f"sidtd/{k}", dataset="sidtd", name=n, doc=d) for k, n, d in _SIDTD},
           **{f"idnet_{k}": dict(key=f"idnet-{k}", dataset="idnet", name=n, doc=d) for k, n, d in _IDNET}}


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _idnet_csv(code, arm):
    p = C_DIR / code / f"fold0_{arm}_q75_predictions.csv"
    return p if p.exists() else C_DIR / "fold0" / f"fold0_{arm}_q75_predictions.csv"   # Arizona lives in fold0/


def _pool():
    """design_id -> family -> list of (image_id, on_row, off_row), sorted by image_id."""
    pool = {}
    on = {r["image_id"]: r for r in _rows(A_DIR / "on/test_predictions.csv")}
    off = {r["image_id"]: r for r in _rows(A_DIR / "off/test_predictions.csv")}
    for i, r in on.items():
        if r["context_available"] == "True":
            d = "sidtd_" + r["design_id"].split("/", 1)[1]
            pool.setdefault(d, {}).setdefault(r["attack_family"] or "genuine", []).append((i, r, off[i]))
    for code, _, _ in _IDNET:
        on = {r["image_id"]: r for r in _rows(_idnet_csv(code, "on")) if r["dataset"] == f"idnet-{code}"}
        off = {r["image_id"]: r for r in _rows(_idnet_csv(code, "off")) if r["dataset"] == f"idnet-{code}"}
        for i, r in on.items():
            if r["available"] == "True":
                pool.setdefault(f"idnet_{code}", {}).setdefault(r["family"], []).append((i, r, off[i]))
    for fams in pool.values():
        for v in fams.values():
            v.sort(key=lambda t: t[0])
    return pool


def eligible_counts() -> dict:
    return {(d, f): len(v) for d, fams in _pool().items() for f, v in fams.items()}


def select_samples(seed: int = 20261015, per_family: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool, out = _pool(), []
    for d, meta in DESIGNS.items():
        for fam in FAMILIES_OF[meta["dataset"]]:
            cands = pool.get(d, {}).get(fam, [])
            k = min(per_family, len(cands))
            if k == 0:
                continue
            for n, j in enumerate(sorted(rng.choice(len(cands), size=k, replace=False)), start=1):
                i, on, off = cands[int(j)]
                out.append(dict(sample_id=f"{d}__{fam}__{n}", image_id=i, design_id=d, design_key=meta["key"],
                                dataset=meta["dataset"], family=fam, label=0 if fam == "genuine" else 1,
                                paper=dict(on=dict(query_logit=float(on["query_logit"]),
                                                   correction_logit=float(on["correction_logit"]),
                                                   attack_logit=float(on["attack_logit"])),
                                           off=dict(attack_logit=float(off["attack_logit"])))))
    return out


def select_mismatch(samples: list[dict]) -> list[dict]:
    first = {s["design_id"]: s for s in samples if s["family"] == "genuine" and s["sample_id"].endswith("__1")}
    return [dict(first[d], mismatch_demo=True) for d in ("sidtd_esp_id", "sidtd_srb_passport", "idnet_AZ")]


if __name__ == "__main__":
    s = select_samples()
    print(len(s), "samples;", len(select_mismatch(s)), "mismatch demos")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/demo/test_select_samples.py -v`
Expected: 3 passed; `python demo/select_samples.py` prints `61 samples; 3 mismatch demos`.

- [ ] **Step 5: Commit**

```bash
git add demo/select_samples.py tests/demo/test_select_samples.py
git commit -m "Demo: seeded sample selection from the locked fold-0 q75 predictions"
```

---

### Task 3: Locate the evaluation requests on the cluster (needs the AUS VPN)

**Files:**
- Create: `demo/sources.json`

**Interfaces:**
- Produces: `demo/sources.json` = `{"a": {"request": {"path", "sha256"}}, "c": {"<CODE>": {"request": {"path", "sha256"}}}}` for codes `FIN, GRC, ESP, ALB, AZ, SD, WV, NV`. "a" is the fold-0 v2noaux Protocol A study request; each "c" entry is the Protocol C v2 score request whose output produced that design's `fold0_*_q75_predictions.csv`.

- [ ] **Step 1: Find the Protocol A fold-0 request**

```bash
ssh b00090279@hpc-gpu.aus.edu
cd /shared/b00090279
for f in $(find . -name test_predictions.csv -path '*evaluation-q75/on*' 2>/dev/null); do echo "$(sha256sum $f | cut -c1-16) $f"; done
```

On the Mac: `shasum -a 256 results/diagnostics/locked-predictions/protocol-a/fold0-v2noaux-fdb7351/evaluation-q75/on/test_predictions.csv | cut -c1-16`. The matching cluster file sits at `<output_root>/evaluation-q75/on/test_predictions.csv`. The request is the JSON whose `"output_root"` equals that `<output_root>`:

```bash
grep -rl "\"output_root\": \"<output_root>\"" /shared/b00090279 --include='*.json' 2>/dev/null
sha256sum <request.json>
```

- [ ] **Step 2: Find the Protocol C requests**

Repeat Step 1 for `evaluation/on/test_predictions.csv` files under `/shared/b00090279/diag-align/c-idnet/` and compare each design's hash with the Mac's `results/diagnostics/protocol-c-idnet/<CODE>/fold0_on_q75_predictions.csv` (Arizona: `results/diagnostics/protocol-c-idnet/fold0/fold0_on_q75_predictions.csv`). If the q75 predictions were produced by a separate re-encoded scorer, its log names the request: `grep -rn "q75" /shared/b00090279/diag-align/c-idnet/*/*.log | head`.

- [ ] **Step 3: Write `demo/sources.json`, check it is complete, and commit**

```bash
python - <<'EOF'
import json
d = json.load(open("demo/sources.json"))
codes = ["FIN", "GRC", "ESP", "ALB", "AZ", "SD", "WV", "NV"]
assert d["a"]["request"]["path"] and len(d["a"]["request"]["sha256"]) == 64
assert all(d["c"][c]["request"]["path"] and len(d["c"][c]["request"]["sha256"]) == 64 for c in codes)
print("sources complete")
EOF
git add demo/sources.json && git commit -m "Demo: cluster paths of the fold-0 A and C evaluation requests"
```

---

### Task 4: Dump the exact network inputs and verify them against the paper (cluster)

**Files:**
- Create: `demo/dump_inputs.py`
- Modify: `.gitignore` (add `demo/out/`)

**Interfaces:**
- Consumes: `select_samples()`, `select_mismatch()` (Task 2); `demo/sources.json` (Task 3); `OnExport`, `OffExport` (Task 1); `scripts/protocol_a_study.py` (`load_request`, `bind`, `bridge`, `completed_arm`, `pinned_gate`); `scripts/score_protocol_c_v2.py` (`load_request`, `authenticate_arms`, `source_inputs`, `driver`); `refer_or_clear.data.reencode.reencoded_queries`; `refer_or_clear.data.protocol_c_reference`.
- Produces: `demo/out/samples/<sample_id>/{query.png, reference.png, masks.png, scores.json}`, `demo/out/designs/<design_id>/{reference.png, zone.png}`, `demo/out/mismatch.json` = `{sample_id: {design_id: "fit" | "no_fit"}}`.
  - `query.png`: 512×800 RGB uint8, the query canvas after q75 re-encoding.
  - `reference.png`: 512×800 RGB uint8, the design median warped into the query frame.
  - `masks.png`: 512×800 RGB uint8, R = zone·255, G = overlap·255, B = 0.
  - `scores.json`: `{"on": {"query_logit", "correction_logit"}, "off": {"query_logit"}}`, recomputed in PyTorch from the PNGs.

- [ ] **Step 1: Write the image helpers**

```python
# demo/dump_inputs.py
"""Dump the exact network inputs of the demo samples from the paper's own evaluation datasets (run on the cluster).

The datasets are constructed as the q75 scorers construct them (protocol_a_study.evaluate_reencoded for SIDTD
fold 0; score_protocol_c_v2.score for IDNet; both inside reencoded_queries(..., 75)). Selected items are written as
lossless PNGs, rescored in PyTorch from those PNGs, and every logit is checked against the paper's locked
predictions (|diff| <= 2e-3). Any mismatch aborts.

Usage (GPU node): python demo/dump_inputs.py --out demo/out
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "demo")]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def to_uint8(chw: np.ndarray) -> np.ndarray:
    """ImageNet-normalized RGB CHW float -> HWC RGB uint8; asserts the tensor came from 8-bit pixels."""
    v = (chw * STD + MEAN) * 255.0
    r = np.rint(v)
    assert np.abs(v - r).max() < 1e-2, "input was not an 8-bit image"
    return np.clip(r, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def from_uint8(hwc: np.ndarray) -> np.ndarray:
    return ((hwc.astype(np.float32).transpose(2, 0, 1) / 255.0) - MEAN) / STD


def write_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
```

- [ ] **Step 2: Add the writer, the PNG round-trip rescoring, and the check against the paper**

```python
def write_item(out: Path, sample_id: str, item: dict):
    """item: one ON dataset item with query (4HW), reference (3HW), overlap (1HW), context_available."""
    assert bool(item["context_available"]), sample_id
    q, r, ov = item["query"].numpy(), item["reference"].numpy(), item["overlap"].numpy()[0]
    d = out / "samples" / sample_id
    write_rgb(d / "query.png", to_uint8(q[:3]))
    write_rgb(d / "reference.png", to_uint8(r))
    masks = np.zeros((512, 800, 3), np.uint8)
    masks[..., 0] = (q[3] > 0.5) * 255
    masks[..., 1] = (ov > 0.5) * 255
    write_rgb(d / "masks.png", masks)


def load_inputs(d: Path):
    m = read_rgb(d / "masks.png")
    zone, ov = (m[..., 0] > 127).astype(np.float32), (m[..., 1] > 127).astype(np.float32)
    q = np.concatenate([from_uint8(read_rgb(d / "query.png")), zone[None]])[None]
    r = from_uint8(read_rgb(d / "reference.png"))[None]
    return torch.from_numpy(q), torch.from_numpy(r), torch.from_numpy(ov[None, None])


def rescore(d: Path, on_export, off_export) -> dict:
    q, r, ov = load_inputs(d)
    with torch.no_grad():
        ql, corr, _ = on_export(q, r, ov)
        oql = off_export(q)
    s = {"on": {"query_logit": float(ql), "correction_logit": float(corr)}, "off": {"query_logit": float(oql)}}
    (d / "scores.json").write_text(json.dumps(s))
    return s


def check(sample: dict, s: dict, tol: float = 2e-3):
    p = sample["paper"]
    errs = {"on_query": abs(s["on"]["query_logit"] - p["on"]["query_logit"]),
            "on_correction": abs(s["on"]["correction_logit"] - p["on"]["correction_logit"]),
            "off": abs(s["off"]["query_logit"] - p["off"]["attack_logit"])}
    bad = {k: v for k, v in errs.items() if v > tol}
    if bad:
        raise SystemExit(f"{sample['sample_id']} differs from the paper: {bad}")
```

- [ ] **Step 3: Add the SIDTD dataset construction (mirrors scripts/protocol_a_study.py:362-378)**

```python
def sidtd_items(binding: dict, image_ids: set) -> dict:
    """image_id -> ON test item of the q75 Protocol A fold-0 evaluation."""
    import protocol_a_study as study
    from refer_or_clear.data.reencode import reencoded_queries
    request, source_context, base, output = study.load_request(binding)
    gate_binding = study.pinned_gate(output, None)
    _, config = study.completed_arm(request, source_context, base, output, "on", gate_binding, binding)
    evaluation_binding = study.bind(output / "evaluation/authorization.json")
    request, context, base, output = study.load_request(binding, evaluation_binding=evaluation_binding)
    table = study.bridge.load_protocol_a_registration(context, study.bind(output / "evaluation/registration.json"),
                                                      roles=["test"])
    items = {}
    with reencoded_queries(study.bridge, request["raw_root"], 75):
        data = study.bridge.ProtocolAReferenceDataset(context, table, role="test", reference_enabled=True,
                                                      augment=dict(config["augment"], enabled=False), cache_bytes=0)
        for k in range(len(data)):
            item = data[k]
            if item["meta"]["image_id"] in image_ids:
                items[item["meta"]["image_id"]] = item
    missing = image_ids - set(items)
    assert not missing, missing
    return items
```

If `study.completed_arm` or `study.pinned_gate` have changed signature, copy the exact calls from `evaluate_reencoded` (scripts/protocol_a_study.py:368-371); the dataset lines must stay identical to lines 375-378.

- [ ] **Step 4: Add the IDNet dataset construction (mirrors scripts/score_protocol_c_v2.py:187-253, reusing the existing evaluation outputs)**

```python
def idnet_items(binding: dict, image_ids: set) -> dict:
    """image_id -> ON test item of the q75 Protocol C evaluation of one IDNet design."""
    import score_protocol_c_v2 as c
    from refer_or_clear.data import protocol_c_reference as consumer
    from refer_or_clear.data.reencode import reencoded_queries
    request = c.load_request(binding)
    study_run = c.authenticate_arms(request)
    records, split = c.source_inputs(request, study_run["request"])
    root = c.driver.absolute(request["output_root"]) / "evaluation"
    context = consumer.load_reference_context(
        records["path"], split["path"], request["target_split"]["path"], records_sha256=records["sha256"],
        target_binding=request["target_enrollment"], enrollment_binding=request["enrollment"],
        raw_root=Path(request["raw_root"]), evaluation_binding=c.driver.bind(root / "authorization.json"),
        root_binding=request["root"])
    table = consumer.load_protocol_c_registration(context, c.driver.bind(root / "registration.json"))
    items = {}
    with reencoded_queries(consumer.direct, request["raw_root"], 75):
        data = consumer.ProtocolCReferenceDataset(context, table, reference_enabled=True)
        for k in range(len(data)):
            item = data[k]
            if item["meta"]["image_id"] in image_ids:
                items[item["meta"]["image_id"]] = item
    missing = image_ids - set(items)
    assert not missing, missing
    return items
```

Before the first run, confirm the module object that the paper's C q75 scoring wrapped: `grep -rn "reencoded_queries(" scripts/ src/`. Use the same module in the `with` line (the module whose `_image` reads query files; for the C consumer this is `protocol_c_reference.direct`).

- [ ] **Step 5: Add design images, the mismatch registration check, and `main`**

```python
def design_images(out: Path, design_id: str, products: dict):
    from refer_or_clear.data import direct_reference as direct
    median = direct._image(products["median"], cv2.IMREAD_COLOR)
    zone = cv2.imread(products["zone"]["path"], cv2.IMREAD_GRAYSCALE)
    d = out / "designs" / design_id
    write_rgb(d / "reference.png", cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    z = np.zeros((512, 800, 3), np.uint8); z[..., 0] = (zone > 127) * 255
    write_rgb(d / "zone.png", z)


def mismatch_status(query_png: Path, banks: dict) -> dict:
    """Registration of one query canvas against each other design's bank with the paper's fit: 'fit' only if the
    similarity fit succeeds and the overlap is usable."""
    from refer_or_clear.data import direct_reference as direct
    from refer_or_clear.data.enrollment import _fit, _similarity
    from refer_or_clear.data.register import _features
    canvas = cv2.imread(str(query_png), cv2.IMREAD_COLOR)
    valid = np.full((512, 800), 255, np.uint8)
    feats, out = _features(canvas), {}
    for design_id, products in banks.items():
        median = direct._image(products["median"], cv2.IMREAD_COLOR)
        _, masks = direct._common(products, registration=False)
        matrix, _ = _fit(feats, _features(median), 12345)
        ok = matrix is not None and direct._available(direct._overlap(valid, masks, _similarity(matrix))[1])
        out[design_id] = "fit" if ok else "no_fit"
    return out


def main(argv=None):
    from bench_cpu_deploy import load_arm
    from export_onnx import OffExport, OnExport
    from select_samples import select_mismatch, select_samples
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo/out")); ap.add_argument("--ckpt", default=str(ROOT / "demo/ckpt"))
    a = ap.parse_args(argv)
    out, ck = Path(a.out), Path(a.ckpt)
    src = json.loads((ROOT / "demo/sources.json").read_text())
    samples = select_samples()
    on = OnExport(load_arm(ck / "on/best.pt", ck / "weights.pth", "on")[0]).eval()
    off = OffExport(load_arm(ck / "off/best.pt", ck / "weights.pth", "off")[0]).eval()
    groups = {"a": [s for s in samples if s["dataset"] == "sidtd"]}
    for code in ("FIN", "GRC", "ESP", "ALB", "AZ", "SD", "WV", "NV"):
        groups[code] = [s for s in samples if s["design_id"] == f"idnet_{code}"]
    banks = {}
    for key, group in groups.items():
        binding = src["a"]["request"] if key == "a" else src["c"][key]["request"]
        items = (sidtd_items if key == "a" else idnet_items)(binding, {s["image_id"] for s in group})
        for s in group:
            item = items[s["image_id"]]
            write_item(out, s["sample_id"], item)
            check(s, rescore(out / "samples" / s["sample_id"], on, off))
            products = item["meta"]["products"]
            if s["design_id"] not in banks:
                banks[s["design_id"]] = products
                design_images(out, s["design_id"], products)
        print(f"{key}: {len(group)} samples verified against the paper")
    mm = {s["sample_id"]: mismatch_status(out / "samples" / s["sample_id"] / "query.png",
                                         {d: b for d, b in banks.items() if d != s["design_id"]})
          for s in select_mismatch(samples)}
    (out / "mismatch.json").write_text(json.dumps(mm, indent=1))
    print("mismatch fits:", {k: sum(v == "fit" for v in m.values()) for k, m in mm.items()})


if __name__ == "__main__":
    main()
```

If `item["meta"]` has no `products` key for the SIDTD bridge dataset, print `sorted(item["meta"])` once and take the bank products from the context instead (`study.bridge._state(context)["banks"][s["design_key"]]["products"]`), keeping everything else unchanged.

- [ ] **Step 6: Run on a GPU node**

```bash
# from the Mac, once
rsync -a demo/ckpt/ b00090279@hpc-gpu.aus.edu:<repo>/demo/ckpt/
# on the cluster, in the repo checkout with the refer-or-clear conda env
git pull
srun --partition=gpu --gres=gpu:1 --time=01:00:00 python demo/dump_inputs.py --out demo/out
```

Expected: nine "verified against the paper" lines (SIDTD 29, each IDNet design 4), `ls demo/out/samples | wc -l` = 61, `ls demo/out/designs | wc -l` = 18, and `mismatch fits` = 0 for all three demos. A "differs from the paper" exit means the inputs are not the paper's: fix the construction, never loosen the tolerance. A non-zero mismatch fit means that document registers to another design; replace that design in `select_mismatch` with another genuine document whose fits are all 0 and rerun.

- [ ] **Step 7: Bring the outputs back and commit the script**

```bash
# on the Mac
rsync -a b00090279@hpc-gpu.aus.edu:<repo>/demo/out/ demo/out/
echo "demo/out/" >> .gitignore
git add .gitignore demo/dump_inputs.py && git commit -m "Demo: dump the paper's exact inputs for the samples, verified against the locked logits"
```

---

### Task 5: Site data and manifest

**Files:**
- Create: `demo/build_site_data.py`
- Test: `tests/demo/test_build_site_data.py`

**Interfaces:**
- Consumes: `demo/out/` (Task 4); `select_samples()`, `select_mismatch()`, `DESIGNS`, `FAMILIES` (Task 2); `site/model/head.json` (Task 1).
- Produces: `site/samples/`, `site/designs/`, and `site/manifest.json`:

```json
{
  "threshold": 0.5,
  "head": {"w_mean": [], "w_max": [], "bias": 0.0},
  "designs": [{"id": "sidtd_aze_passport", "name": "Azerbaijan passport", "dataset": "SIDTD",
               "reference": "designs/sidtd_aze_passport/reference.png", "zone": "designs/sidtd_aze_passport/zone.png"}],
  "samples": [{"id": "sidtd_aze_passport__text_inpaint__1", "design": "sidtd_aze_passport", "family": "text_inpaint",
               "family_name": "Inpaint-and-rewrite", "n": 1, "label": 1,
               "query": "samples/sidtd_aze_passport__text_inpaint__1/query.png",
               "reference": "samples/sidtd_aze_passport__text_inpaint__1/reference.png",
               "masks": "samples/sidtd_aze_passport__text_inpaint__1/masks.png",
               "pytorch": {"on": {"p": 0.97}, "off": {"p": 0.31}}}],
  "mismatch": [{"sample": "sidtd_esp_id__genuine__1", "label": "Spain identity card · genuine", "fits": {"sidtd_aze_passport": false}}]
}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/demo/test_build_site_data.py
import csv, json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
SITE = ROOT / "site"
pytestmark = pytest.mark.skipif(not (SITE / "manifest.json").exists(), reason="run demo/build_site_data.py first")


def load():
    return json.loads((SITE / "manifest.json").read_text())


def test_every_file_exists():
    m = load()
    for d in m["designs"]:
        assert (SITE / d["reference"]).exists() and (SITE / d["zone"]).exists()
    for s in m["samples"]:
        for k in ("query", "reference", "masks"):
            assert (SITE / s[k]).exists(), s[k]


def test_counts_names_and_wording():
    m = load()
    assert len(m["designs"]) == 18 and len(m["samples"]) == 61 and len(m["mismatch"]) == 3
    assert "enrol" not in json.dumps(m).lower()
    assert {s["family_name"] for s in m["samples"]} == {"Genuine", "Inpaint-and-rewrite", "Copy-and-replace",
                                                       "In-place text replacement"}


def test_sidtd_samples_are_test_documents():
    from select_samples import select_samples
    split = ROOT / "data/processed/splits/protocol_a_it0.csv"
    if not split.exists():
        pytest.skip("SIDTD split not present")
    roles = {r["image_id"]: r["role"] for r in csv.DictReader(open(split))}
    for s in select_samples():
        if s["dataset"] == "sidtd":
            assert roles[s["image_id"]] == "test", s["image_id"]


def test_pytorch_scores_present_and_in_range():
    for s in load()["samples"]:
        assert 0.0 <= s["pytorch"]["on"]["p"] <= 1.0 and 0.0 <= s["pytorch"]["off"]["p"] <= 1.0
```

- [ ] **Step 2: Run test to verify it skips**

Run: `pytest tests/demo/test_build_site_data.py -v`
Expected: 4 skipped (`run demo/build_site_data.py first`).

- [ ] **Step 3: Write the implementation**

```python
# demo/build_site_data.py
"""Copy the dumped inputs into site/ and write site/manifest.json. Usage: python demo/build_site_data.py"""
from __future__ import annotations

import json, math, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from select_samples import DESIGNS, FAMILIES, select_mismatch, select_samples  # noqa: E402

OUT, SITE = ROOT / "demo/out", ROOT / "site"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main():
    samples = select_samples()
    for sub in ("samples", "designs"):
        if (SITE / sub).exists():
            shutil.rmtree(SITE / sub)
        shutil.copytree(OUT / sub, SITE / sub)
    designs = [dict(id=d, name=f"{m['name']} {m['doc']}", dataset="SIDTD" if m["dataset"] == "sidtd" else "IDNet",
                    reference=f"designs/{d}/reference.png", zone=f"designs/{d}/zone.png") for d, m in DESIGNS.items()]
    rows = []
    for s in samples:
        sc = json.loads((OUT / "samples" / s["sample_id"] / "scores.json").read_text())
        rows.append(dict(id=s["sample_id"], design=s["design_id"], family=s["family"], family_name=FAMILIES[s["family"]],
                         n=int(s["sample_id"].rsplit("__", 1)[1]), label=s["label"],
                         query=f"samples/{s['sample_id']}/query.png", reference=f"samples/{s['sample_id']}/reference.png",
                         masks=f"samples/{s['sample_id']}/masks.png",
                         pytorch=dict(on=dict(p=sigmoid(sc["on"]["query_logit"] + sc["on"]["correction_logit"])),
                                      off=dict(p=sigmoid(sc["off"]["query_logit"])))))
    fits = json.loads((OUT / "mismatch.json").read_text())
    mismatch = [dict(sample=s["sample_id"],
                     label=f"{DESIGNS[s['design_id']]['name']} {DESIGNS[s['design_id']]['doc']} · genuine",
                     fits={d: v == "fit" for d, v in fits[s["sample_id"]].items()}) for s in select_mismatch(samples)]
    head = json.loads((SITE / "model/head.json").read_text())
    (SITE / "manifest.json").write_text(json.dumps(dict(threshold=0.5, head=head, designs=designs, samples=rows,
                                                        mismatch=mismatch), indent=1))
    print(len(designs), "designs,", len(rows), "samples,", len(mismatch), "mismatch demos")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it and the tests**

Run: `python demo/build_site_data.py && pytest tests/demo/test_build_site_data.py -v`
Expected: `18 designs, 61 samples, 3 mismatch demos`; 4 passed.

- [ ] **Step 5: Commit**

```bash
du -sh site/samples site/designs
git add demo/build_site_data.py tests/demo/test_build_site_data.py site/manifest.json site/samples site/designs
git commit -m "Demo: site data and manifest (61 samples, 18 design references)"
```

---

### Task 6: Pure browser modules (tensors, decision, heat map, library)

**Files:**
- Create: `site/js/tensors.js`, `site/js/decision.js`, `site/js/heatmap.js`, `site/js/library.js`
- Test: `site/tests/tensors.test.mjs`, `site/tests/decision.test.mjs`, `site/tests/heatmap.test.mjs`, `site/tests/library.test.mjs`

**Interfaces:**
- Produces:
  - `tensors.js`: `rgbToNormalizedCHW(rgba: Uint8Array, w, h) -> Float32Array(3*h*w)`; `buildQuery(queryRgba, masksRgba, w, h) -> Float32Array(4*h*w)`; `buildOverlap(masksRgba, w, h) -> Float32Array(h*w)`; `decodePng(arrayBuffer, UPNG) -> {rgba: Uint8Array, width, height}`.
  - `decision.js`: `sigmoid(x)`; `decide(p, threshold = 0.5) -> "REFER" | "CLEAR"`; `decideMismatch() -> {decision: "REFER", reason: "no reference context, sent to the second line"}`; `percent(p) -> string`.
  - `heatmap.js`: `differenceMap(local: Float32Array(32*64*100), overlapSmall: Uint8Array(6400), head) -> Float32Array(6400)` in [0, 1]; `downsampleOverlap(overlap: Float32Array(512*800)) -> Uint8Array(6400)`.
  - `library.js`: `documentOptions(manifest, designId) -> [{group, items: [{value, text}]}]`; `findSample(manifest, value) -> {kind: "sample", sample} | {kind: "mismatch", entry, sample}`.

- [ ] **Step 1: Write the failing tests**

```js
// site/tests/tensors.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { rgbToNormalizedCHW, buildQuery, buildOverlap } from "../js/tensors.js";

test("normalizes like the paper (ImageNet mean/std, RGB order, CHW)", () => {
  const rgba = new Uint8Array([255, 0, 128, 255, 0, 255, 64, 255]); // w=2, h=1
  const t = rgbToNormalizedCHW(rgba, 2, 1);
  assert.equal(t.length, 6);
  assert.ok(Math.abs(t[0] - (1 - 0.485) / 0.229) < 1e-6);
  assert.ok(Math.abs(t[1] - (0 - 0.485) / 0.229) < 1e-6);
  assert.ok(Math.abs(t[2] - (0 - 0.456) / 0.224) < 1e-6);
  assert.ok(Math.abs(t[5] - (64 / 255 - 0.406) / 0.225) < 1e-6);
});

test("query takes the zone from R as its 4th channel; overlap comes from G", () => {
  const t = buildQuery(new Uint8Array([10, 20, 30, 255]), new Uint8Array([255, 0, 0, 255]), 1, 1);
  assert.equal(t.length, 4); assert.equal(t[3], 1);
  assert.deepEqual(Array.from(buildOverlap(new Uint8Array([255, 0, 0, 255]), 1, 1)), [0]);
  assert.deepEqual(Array.from(buildOverlap(new Uint8Array([0, 255, 0, 255]), 1, 1)), [1]);
});
```

```js
// site/tests/decision.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { sigmoid, decide, decideMismatch, percent } from "../js/decision.js";

test("refers above 0.5 and clears at or below (ties genuine, as in the paper)", () => {
  assert.equal(decide(0.51), "REFER"); assert.equal(decide(0.5), "CLEAR"); assert.equal(decide(0.2), "CLEAR");
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
});
test("a document of another design is always referred", () => {
  assert.deepEqual(decideMismatch(), { decision: "REFER", reason: "no reference context, sent to the second line" });
});
test("percent formatting", () => {
  assert.equal(percent(0.968), "97%"); assert.equal(percent(0.002), "<1%"); assert.equal(percent(0.9995), ">99%");
});
```

```js
// site/tests/heatmap.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { differenceMap, downsampleOverlap } from "../js/heatmap.js";

test("head-weighted channel sum, zero outside the overlap, scaled to [0,1]", () => {
  const N = 6400, local = new Float32Array(32 * N);
  local[5] = 2; local[N + 7] = 1;                    // channel 0 at pixel 5, channel 1 at pixel 7
  const ov = new Uint8Array(N); ov[5] = 1; ov[7] = 1;
  const head = { w_mean: Array(32).fill(0), w_max: Array(32).fill(0), bias: 0 };
  head.w_max[0] = 1; head.w_max[1] = 1;
  const m = differenceMap(local, ov, head);
  assert.equal(m[5], 1); assert.equal(m[7], 0.5); assert.equal(m[6], 0);
  ov[5] = 0; assert.equal(differenceMap(local, ov, head)[5], 0);
});

test("overlap downsampling is nearest with stride 8", () => {
  const big = new Float32Array(512 * 800); big[8 * 800 + 16] = 1;
  const small = downsampleOverlap(big);
  assert.equal(small.length, 6400); assert.equal(small[100 + 2], 1); assert.equal(small[0], 0);
});
```

```js
// site/tests/library.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { documentOptions, findSample } from "../js/library.js";

const manifest = {
  designs: [{ id: "a", name: "A passport" }, { id: "b", name: "B card" }],
  samples: [
    { id: "a__genuine__1", design: "a", family: "genuine", family_name: "Genuine", n: 1, label: 0 },
    { id: "a__text_inpaint__1", design: "a", family: "text_inpaint", family_name: "Inpaint-and-rewrite", n: 1, label: 1 },
    { id: "b__genuine__1", design: "b", family: "genuine", family_name: "Genuine", n: 1, label: 0 }],
  mismatch: [{ sample: "b__genuine__1", label: "B card · genuine", fits: { a: false } },
             { sample: "a__genuine__1", label: "A passport · genuine", fits: { b: false } }],
};

test("options list the chosen design, then documents of other designs only", () => {
  const g = documentOptions(manifest, "a");
  assert.equal(g[0].group, "A passport");
  assert.deepEqual(g[0].items.map((i) => i.text), ["Genuine · #1", "Inpaint-and-rewrite · #1"]);
  assert.equal(g[1].group, "Another design (no reference match)");
  assert.deepEqual(g[1].items.map((i) => i.text), ["B card · genuine"]);
});

test("findSample resolves both kinds", () => {
  assert.equal(findSample(manifest, "a__text_inpaint__1").kind, "sample");
  const m = findSample(manifest, "mismatch:b__genuine__1");
  assert.equal(m.kind, "mismatch"); assert.equal(m.sample.id, "b__genuine__1");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `node --test site/tests/`
Expected: FAIL with `Cannot find module` for `../js/tensors.js` (and the other three).

- [ ] **Step 3: Write the modules**

```js
// site/js/tensors.js
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export function rgbToNormalizedCHW(rgba, w, h) {
  const n = w * h, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) out[c * n + i] = (rgba[4 * i + c] / 255 - MEAN[c]) / STD[c];
  return out;
}

export function buildQuery(queryRgba, masksRgba, w, h) {
  const n = w * h, out = new Float32Array(4 * n);
  out.set(rgbToNormalizedCHW(queryRgba, w, h), 0);
  for (let i = 0; i < n; i++) out[3 * n + i] = masksRgba[4 * i] > 127 ? 1 : 0;
  return out;
}

export function buildOverlap(masksRgba, w, h) {
  const n = w * h, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = masksRgba[4 * i + 1] > 127 ? 1 : 0;
  return out;
}

/** Exact PNG decode with UPNG (no canvas, so no color management or fingerprint noise): 8-bit RGBA. */
export function decodePng(buffer, UPNG) {
  const img = UPNG.decode(buffer);
  return { rgba: new Uint8Array(UPNG.toRGBA8(img)[0]), width: img.width, height: img.height };
}
```

```js
// site/js/decision.js
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export const decide = (p, threshold = 0.5) => (p > threshold ? "REFER" : "CLEAR");
export const decideMismatch = () => ({ decision: "REFER", reason: "no reference context, sent to the second line" });

export function percent(p) {
  if (p < 0.005) return "<1%";
  if (p > 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}
```

```js
// site/js/heatmap.js
const C = 32, H = 64, W = 100, N = H * W;

/** M = sum_c (w_mean_c / |overlap| + w_max_c) F_c inside the overlap, clipped at 0, scaled by its maximum. */
export function differenceMap(local, overlapSmall, head) {
  let area = 0;
  for (let i = 0; i < N; i++) area += overlapSmall[i] ? 1 : 0;
  const w = head.w_mean.map((m, c) => m / Math.max(area, 1) + head.w_max[c]);
  const out = new Float32Array(N);
  let max = 0;
  for (let i = 0; i < N; i++) {
    if (!overlapSmall[i]) continue;
    let s = 0;
    for (let c = 0; c < C; c++) s += w[c] * local[c * N + i];
    out[i] = Math.max(s, 0);
    if (out[i] > max) max = out[i];
  }
  if (max > 0) for (let i = 0; i < N; i++) out[i] /= max;
  return out;
}

export function downsampleOverlap(overlap) {
  const out = new Uint8Array(N);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) out[y * W + x] = overlap[y * 8 * 800 + x * 8] > 0.5 ? 1 : 0;
  return out;
}
```

```js
// site/js/library.js
export function documentOptions(manifest, designId) {
  const design = manifest.designs.find((d) => d.id === designId);
  const own = manifest.samples
    .filter((s) => s.design === designId)
    .map((s) => ({ value: s.id, text: `${s.family_name} · #${s.n}` }));
  const others = manifest.mismatch
    .filter((m) => !m.sample.startsWith(`${designId}__`))
    .map((m) => ({ value: `mismatch:${m.sample}`, text: m.label }));
  return [{ group: design.name, items: own }, { group: "Another design (no reference match)", items: others }];
}

export function findSample(manifest, value) {
  if (value.startsWith("mismatch:")) {
    const id = value.slice("mismatch:".length);
    return { kind: "mismatch", entry: manifest.mismatch.find((m) => m.sample === id),
             sample: manifest.samples.find((s) => s.id === id) };
  }
  return { kind: "sample", sample: manifest.samples.find((s) => s.id === value) };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `node --test site/tests/`
Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add site/js/tensors.js site/js/decision.js site/js/heatmap.js site/js/library.js site/tests
git commit -m "Demo site: tensor assembly, decision, difference map and library modules with node tests"
```

---

### Task 7: Runner and the page (storyboard, toggle, schematic, result)

**Files:**
- Create: `site/js/runner.js`, `site/js/main.js`, `site/index.html`, `site/style.css`
- Test: `site/tests/runner.test.mjs`

**Interfaces:**
- Consumes: Task 6 modules; `site/manifest.json`; `site/model/*.onnx`; globals `ort` (onnxruntime-web 1.20.1), `UPNG`, `pako`.
- Produces: `pickProviders(nav) -> string[]`; class `RunGate` (`tryEnter() -> boolean`, `leave()`); `loadSession(ort, url, providers, onProgress) -> Promise<{session, backend}>`; `runOnce(session, feeds) -> Promise<{outputs, ms}>`.

- [ ] **Step 1: Write the failing test**

```js
// site/tests/runner.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { pickProviders, RunGate } from "../js/runner.js";

test("WebGPU first when available, WebAssembly otherwise", () => {
  assert.deepEqual(pickProviders({ gpu: {} }), ["webgpu", "wasm"]);
  assert.deepEqual(pickProviders({}), ["wasm"]);
});

test("one run at a time", () => {
  const g = new RunGate();
  assert.equal(g.tryEnter(), true); assert.equal(g.tryEnter(), false);
  g.leave(); assert.equal(g.tryEnter(), true);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test site/tests/runner.test.mjs`
Expected: FAIL with `Cannot find module` for `../js/runner.js`.

- [ ] **Step 3: Write `runner.js`**

```js
// site/js/runner.js
export const pickProviders = (nav) => (nav && nav.gpu ? ["webgpu", "wasm"] : ["wasm"]);

export class RunGate {
  #busy = false;
  tryEnter() { if (this.#busy) return false; this.#busy = true; return true; }
  leave() { this.#busy = false; }
}

/** Download with progress, then create the session with the first provider that works; reports which one. */
export async function loadSession(ort, url, providers, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(total ? got / total : 0);
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  for (const p of providers) {
    try { return { session: await ort.InferenceSession.create(bytes, { executionProviders: [p] }), backend: p }; }
    catch (e) { console.warn(`provider ${p} failed`, e); }
  }
  throw new Error("no execution provider could load the model");
}

export async function runOnce(session, feeds) {
  const t0 = performance.now();
  const outputs = await session.run(feeds);
  return { outputs, ms: performance.now() - t0 };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test site/tests/`
Expected: 11 tests pass.

- [ ] **Step 5: Write `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReferNet Demo</title>
<link rel="stylesheet" href="style.css">
<script src="https://cdn.jsdelivr.net/npm/pako@1.0.11/dist/pako.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/upng-js@2.1.0/UPNG.js"></script>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js"></script>
<script type="module" src="js/main.js"></script>
</head>
<body>
<header>
  <div>
    <h1>ReferNet <span>· reference-assisted document forgery detection</span></h1>
    <p>Pick a design and a document. The paper's deployment model runs in your browser, with or without the design reference.</p>
  </div>
  <nav><a href="https://github.com/7abushahla/ReferNet">Code</a><a href="#how">How it works ↓</a></nav>
</header>
<main>
  <section class="story">
    <div class="card">
      <h2><span class="n">1</span>Design reference</h2>
      <select id="design" aria-label="Design reference"></select>
      <div class="frame" id="ref-frame"><img id="ref-img" alt="Design reference"><img id="zone-img" class="zone" alt=""></div>
      <p class="hint">Median of genuine documents · hover to see the personal-data zone</p>
    </div>
    <div class="arrow" aria-hidden="true">→</div>
    <div class="card">
      <h2><span class="n">2</span>Presented document</h2>
      <select id="document" aria-label="Presented document"></select>
      <div class="frame"><img id="doc-img" alt="Presented document"><canvas id="heat"></canvas></div>
      <p class="hint" id="truth-hint">Ground truth hidden until the run</p>
    </div>
    <div class="arrow" aria-hidden="true">→</div>
    <div class="card">
      <h2><span class="n">3</span>ReferNet</h2>
      <svg id="schematic" viewBox="0 0 230 250" role="img" aria-label="ReferNet schematic">
        <defs>
          <marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#1f4e79"/></marker>
          <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#c8102e"/></marker>
        </defs>
        <rect x="8" y="66" width="214" height="30" rx="5" class="enc"/>
        <text x="115" y="79" class="t1">Shared encoder, early stages</text><text x="115" y="90" class="t2">same weights for both inputs</text>
        <g>
          <rect x="8" y="8" width="88" height="34" rx="5" class="qbox"/><text x="52" y="23" class="t1 qt">Document</text><text x="52" y="35" class="t2">+ data zone</text>
          <line x1="52" y1="42" x2="52" y2="64" class="ql flow" marker-end="url(#ab)"/>
          <rect x="8" y="122" width="88" height="32" rx="5" class="qbox"/><text x="52" y="136" class="t1 qt">Deeper stages</text><text x="52" y="147" class="t2">judge alone</text>
          <line x1="52" y1="96" x2="52" y2="120" class="ql flow" marker-end="url(#ab)"/>
          <line x1="52" y1="154" x2="103" y2="184" class="ql flow" marker-end="url(#ab)"/><text x="60" y="176" class="t2 qt">score</text>
        </g>
        <g class="r">
          <rect x="134" y="8" width="88" height="34" rx="5" class="rbox"/><text x="178" y="23" class="t1 rt">Design reference</text><text x="178" y="35" class="t2">median of genuine</text>
          <line x1="178" y1="42" x2="178" y2="64" class="rl flow" marker-end="url(#ar)"/>
          <rect x="134" y="122" width="88" height="32" rx="5" class="rbox"/><text x="178" y="136" class="t1 rt">Compare</text><text x="178" y="147" class="t2">document vs reference</text>
          <line x1="178" y1="96" x2="178" y2="120" class="rl flow" marker-end="url(#ar)"/>
          <line x1="110" y1="96" x2="150" y2="120" class="ql flow" marker-end="url(#ab)" opacity=".7"/>
          <line x1="178" y1="154" x2="127" y2="184" class="rl flow" marker-end="url(#ar)"/><text x="150" y="176" class="t2 rt">correction</text>
        </g>
        <circle cx="115" cy="190" r="13" class="plus"/><text x="115" y="195" class="t3">+</text>
        <line x1="115" y1="203" x2="115" y2="218" class="out"/>
        <rect x="70" y="220" width="90" height="24" rx="12" class="decide"/><text x="115" y="236" class="t1 w">Refer / Clear</text>
      </svg>
      <div class="seg" role="radiogroup" aria-label="Condition">
        <button id="with" class="on" role="radio" aria-checked="true">With reference</button>
        <button id="without" role="radio" aria-checked="false">Without reference</button>
      </div>
      <button id="run" disabled>Loading model…</button>
      <div class="progress"><div id="bar"></div></div>
      <p class="hint" id="backend"></p>
    </div>
    <div class="arrow" aria-hidden="true">→</div>
    <div class="card">
      <h2><span class="n">4</span>Decision</h2>
      <div id="result" class="empty" aria-live="polite">Run ReferNet to see its decision.</div>
    </div>
  </section>

  <section id="how" class="how">
    <h2>How it works</h2>
    <p>A design reference is the pixel-wise median of genuine documents of one design, registered to each other. Personal data differ from document to document and fade in the median, while the printed design stays sharp. The personal-data zone marks where a document is allowed to differ from its design.</p>
    <p>ReferNet judges the presented document on its own, through a single-image path, and, with the reference, compares it with the design reference through a second path that shares the same encoder. The comparison produces a correction that can raise or lower the single-image decision. Without the reference, ReferNet is a single-image classifier like the published baselines.</p>
    <p>In this demo, each document has already been registered to its design reference with the paper's code, and its compression equalized as in the paper. The network itself runs live in your browser. Inference times are measured on your device, whereas the paper's 280 ms is end to end on a laptop processor.</p>
    <div class="note"><b>About the difference map.</b> The map in step 4 is not part of the paper. It is shown for demonstration only. It is computed in your browser from the same run, from the comparison features weighted by the correction layer, to indicate which region drove the reference's correction, and it has not been evaluated as a localization method.</div>
  </section>
</main>
<footer>Samples from SIDTD (CC BY 4.0; MIDV-2020 images CC BY-SA 2.5) and IDNet (CC0) · registration precomputed with the paper's code · fold-0 deployment model, FP32 · scores verified against PyTorch (<a href="verify.html">check</a>)</footer>
</body>
</html>
```

- [ ] **Step 6: Write `style.css`**

```css
:root { --ink:#1d2733; --muted:#6b7785; --line:#d6dde6; --bg:#f6f8fb; --card:#fff; --blue:#1f4e79; --red:#c8102e; --green:#1b7f3b; --head:#0f2740; --qfill:#eef4fb; --rfill:#fdf0f2; --efill:#f4f6f9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --ink:#e8eef5; --muted:#9fb0c3; --line:#2b3d52; --bg:#0b1522; --card:#132235; --head:#07101b; --qfill:#16314d; --rfill:#3a1a22; --efill:#1b2b3f; } }
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
header { background:var(--head); color:#fff; padding:16px 20px; display:flex; justify-content:space-between; gap:16px; align-items:center; flex-wrap:wrap; }
header h1 { margin:0; font-size:20px; } header h1 span { font-weight:400; color:#cfe0f2; font-size:15px; }
header p { margin:4px 0 0; color:#9fb6cf; font-size:13px; } header nav a { color:#cfe0f2; margin-left:16px; text-decoration:none; font-size:13px; }
main { padding:16px 20px; max-width:1400px; margin:0 auto; }
.story { display:flex; gap:10px; align-items:stretch; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; flex:1; display:flex; flex-direction:column; gap:8px; min-width:0; }
.card h2 { font-size:15px; margin:0; display:flex; align-items:center; }
.n { background:var(--blue); color:#fff; border-radius:50%; width:22px; height:22px; display:inline-flex; align-items:center; justify-content:center; font-size:12px; margin-right:8px; }
.arrow { align-self:center; font-size:24px; color:var(--muted); }
select { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); font-size:13px; }
.frame { position:relative; } .frame img, .frame canvas { width:100%; display:block; border-radius:4px; }
.frame .zone, .frame canvas { position:absolute; inset:0; height:100%; pointer-events:none; }
.frame .zone { opacity:0; transition:opacity .2s; mix-blend-mode:multiply; } #ref-frame:hover .zone { opacity:.55; }
.hint { font-size:12px; color:var(--muted); margin:0; }
#schematic { width:100%; max-height:240px; }
#schematic text { text-anchor:middle; } .t1 { font-size:9.5px; font-weight:600; fill:var(--ink); } .t2 { font-size:8.5px; fill:var(--muted); } .t3 { font-size:15px; font-weight:700; fill:var(--ink); }
.qbox { fill:var(--qfill); stroke:var(--blue); } .rbox { fill:var(--rfill); stroke:var(--red); } .enc { fill:var(--efill); stroke:#8a96a6; } .qt { fill:var(--blue); } .rt { fill:var(--red); }
.ql { stroke:var(--blue); stroke-width:1.6; } .rl { stroke:var(--red); stroke-width:1.6; } .plus { fill:var(--card); stroke:var(--ink); stroke-width:1.4; } .out { stroke:var(--ink); stroke-width:1.6; } .decide { fill:#1d2733; } .w { fill:#fff; }
#schematic.off .r { opacity:.18; filter:grayscale(1); }
#schematic.running .flow { stroke-dasharray:5 5; animation:flow .8s linear infinite; } @keyframes flow { to { stroke-dashoffset:-20; } }
.seg { display:flex; border:1px solid var(--blue); border-radius:6px; overflow:hidden; }
.seg button { flex:1; border:0; padding:7px; background:transparent; color:var(--blue); font-size:13px; cursor:pointer; } .seg button.on { background:var(--blue); color:#fff; font-weight:600; }
#run { background:var(--blue); color:#fff; border:0; border-radius:6px; padding:9px; font-weight:600; font-size:14px; cursor:pointer; } #run:disabled { opacity:.5; cursor:default; }
.progress { height:4px; background:var(--line); border-radius:2px; overflow:hidden; } #bar { height:100%; width:0; background:var(--blue); transition:width .2s; }
#result.empty { color:var(--muted); font-size:13px; }
.pct { font-size:34px; font-weight:800; line-height:1; } .pct small { font-size:12px; font-weight:400; color:var(--muted); display:block; margin-top:2px; }
.verdict { color:#fff; font-weight:800; text-align:center; border-radius:6px; padding:9px; font-size:19px; letter-spacing:.5px; } .REFER { background:var(--red); } .CLEAR { background:var(--green); }
.kv { display:flex; justify-content:space-between; gap:8px; font-size:13px; } .ok { color:var(--green); } .bad { color:var(--red); }
.how { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-top:16px; font-size:14px; line-height:1.55; }
.how h2 { font-size:16px; margin:0 0 6px; } .note { background:#fff7e6; color:#3b2f12; border:1px solid #f0d9a8; border-radius:6px; padding:8px 10px; }
footer { padding:12px 20px; font-size:12px; color:var(--muted); border-top:1px solid var(--line); } footer a { color:inherit; }
@media (max-width: 900px) { .story { flex-direction:column; } .arrow { transform:rotate(90deg); } main, header, footer { padding-left:16px; padding-right:16px; } }
```

- [ ] **Step 7: Write `main.js`**

```js
// site/js/main.js
import { buildQuery, buildOverlap, rgbToNormalizedCHW, decodePng } from "./tensors.js";
import { sigmoid, decide, decideMismatch, percent } from "./decision.js";
import { differenceMap, downsampleOverlap } from "./heatmap.js";
import { documentOptions, findSample } from "./library.js";
import { pickProviders, RunGate, loadSession, runOnce } from "./runner.js";

const W = 800, H = 512, ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const $ = (id) => document.getElementById(id);
const state = { manifest: null, sessions: {}, loading: {}, withRef: true, gate: new RunGate() };

async function png(url) { return decodePng(await (await fetch(url)).arrayBuffer(), window.UPNG); }

function fillDesigns() {
  for (const d of state.manifest.designs) $("design").add(new Option(`${d.name} · ${d.dataset}`, d.id));
  $("design").onchange = onDesign;
  onDesign();
}

function onDesign() {
  const d = state.manifest.designs.find((x) => x.id === $("design").value);
  $("ref-img").src = d.reference; $("zone-img").src = d.zone;
  const sel = $("document"); sel.innerHTML = "";
  for (const g of documentOptions(state.manifest, d.id)) {
    const og = document.createElement("optgroup"); og.label = g.group;
    for (const it of g.items) og.appendChild(new Option(it.text, it.value));
    sel.appendChild(og);
  }
  sel.onchange = onDocument;
  onDocument();
}

function onDocument() {
  $("doc-img").src = findSample(state.manifest, $("document").value).sample.query;
  clearResult();
}

function clearResult() {
  $("result").className = "empty"; $("result").textContent = "Run ReferNet to see its decision.";
  $("truth-hint").textContent = "Ground truth hidden until the run";
  const c = $("heat"); c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function setCondition(withRef) {
  state.withRef = withRef;
  $("with").classList.toggle("on", withRef); $("without").classList.toggle("on", !withRef);
  $("with").setAttribute("aria-checked", String(withRef)); $("without").setAttribute("aria-checked", String(!withRef));
  $("schematic").classList.toggle("off", !withRef);
  clearResult();
  if (!withRef) ensureSession("off");
}

function feeds(arm, q, r, ov) {
  const T = window.ort.Tensor, f = { query: new T("float32", q, [1, 4, H, W]) };
  if (arm === "on") { f.reference = new T("float32", r, [1, 3, H, W]); f.overlap = new T("float32", ov, [1, 1, H, W]); }
  return f;
}

function ensureSession(arm) {
  if (!state.loading[arm]) {
    state.loading[arm] = (async () => {
      $("run").disabled = true; $("run").textContent = "Loading model…"; $("bar").style.width = "0";
      const { session, backend } = await loadSession(window.ort, `model/refernet_${arm}.onnx`, pickProviders(navigator),
        (f) => ($("bar").style.width = `${Math.round(f * 100)}%`));
      await runOnce(session, feeds(arm, new Float32Array(4 * H * W), new Float32Array(3 * H * W), new Float32Array(H * W).fill(1)));
      state.sessions[arm] = session;
      $("backend").textContent = `runs in your browser · ${backend === "webgpu" ? "WebGPU" : "WebAssembly"}`;
      $("run").disabled = false; $("run").textContent = "▶ Run";
      return session;
    })();
  }
  return state.loading[arm];
}

function drawHeat(map) {
  const c = $("heat"); c.width = 100; c.height = 64;
  const ctx = c.getContext("2d"), img = ctx.createImageData(100, 64);
  for (let i = 0; i < 6400; i++) {
    const v = map[i];
    img.data[4 * i] = 255; img.data[4 * i + 1] = Math.round(220 * (1 - v)); img.data[4 * i + 2] = 0;
    img.data[4 * i + 3] = v > 0.15 ? Math.round(170 * v) : 0;
  }
  ctx.putImageData(img, 0, 0);
}

function showResult({ p, ms, label, mismatch }) {
  const truth = label === 1 ? "altered" : "genuine";
  $("result").className = "";
  if (mismatch) {
    $("result").innerHTML = `<div class="verdict REFER">REFER</div><p class="hint">${decideMismatch().reason}</p>
      <div class="kv"><span>Ground truth</span><span>${truth} · document of another design</span></div>`;
  } else {
    const d = decide(p, state.manifest.threshold), correct = (d === "REFER") === (label === 1);
    $("result").innerHTML = `<p class="hint">${state.withRef ? "With reference" : "Without reference"}</p>
      <div class="pct" style="color:var(--${d === "REFER" ? "red" : "green"})">${percent(p)}<small>likelihood of alteration</small></div>
      <div class="verdict ${d}">${d}</div>
      <div class="kv"><span>Inference time</span><b>${Math.round(ms)} ms</b></div>
      <div class="kv"><span>Ground truth</span><span>${truth} · <span class="${correct ? "ok" : "bad"}">${correct ? "correct ✓" : "wrong ✗"}</span></span></div>`;
  }
  $("truth-hint").textContent = `Ground truth: ${truth}`;
}

async function onRun() {
  if (!state.gate.tryEnter()) return;
  $("run").disabled = true; $("schematic").classList.add("running");
  try {
    const f = findSample(state.manifest, $("document").value);
    if (f.kind === "mismatch") { showResult({ label: f.sample.label, mismatch: true }); return; }
    const s = f.sample, arm = state.withRef ? "on" : "off";
    const session = await ensureSession(arm);
    const [q, m, r] = await Promise.all([png(s.query), png(s.masks), arm === "on" ? png(s.reference) : null]);
    const qt = buildQuery(q.rgba, m.rgba, W, H), ov = buildOverlap(m.rgba, W, H);
    const rt = r ? rgbToNormalizedCHW(r.rgba, W, H) : null;
    const { outputs, ms } = await runOnce(session, feeds(arm, qt, rt, ov));
    const logit = outputs.query_logit.data[0] + (arm === "on" ? outputs.correction.data[0] : 0);
    showResult({ p: sigmoid(logit), ms, label: s.label });
    if (arm === "on") drawHeat(differenceMap(outputs.local.data, downsampleOverlap(ov), state.manifest.head));
  } catch (e) {
    $("result").className = "";
    $("result").textContent = `This browser could not run the model (${e.message}). Try a recent Chrome, Edge, Firefox or Safari.`;
  } finally {
    $("schematic").classList.remove("running"); $("run").disabled = false; state.gate.leave();
  }
}

async function init() {
  if (!window.ort || !window.UPNG) {
    $("result").className = ""; $("result").textContent = "The model runtime could not load. Try a recent Chrome, Edge, Firefox or Safari.";
    return;
  }
  window.ort.env.wasm.wasmPaths = ORT_DIST;
  state.manifest = await (await fetch("manifest.json")).json();
  fillDesigns();
  $("with").onclick = () => setCondition(true);
  $("without").onclick = () => setCondition(false);
  $("run").onclick = onRun;
  await ensureSession("on");
}

init();
```

- [ ] **Step 8: Check it in browsers**

```bash
cd site && python3 -m http.server 8765
```

Open `http://localhost:8765/` in Chrome, then Safari and Firefox:
1. The progress bar fills, the button becomes "▶ Run", and the backend line says WebGPU (Chrome) or WebAssembly (Safari/Firefox without WebGPU).
2. Azerbaijan passport · Inpaint-and-rewrite · #1 · With reference · Run: percentage, verdict, time, ground truth; the heat map overlays the document. The percentage equals `pytorch.on.p` of that sample in `manifest.json`, rounded.
3. Without reference: the red branch greys out, the OFF model downloads once with the bar, Run gives no heat map, and the percentage equals `pytorch.off.p`.
4. An "Another design (no reference match)" item: REFER with "no reference context, sent to the second line", for both toggle states, and no inference time.
5. Double-click Run quickly: one result; the button stays disabled during the run.
6. DevTools → Network → Slow 4G, reload: Run stays disabled until the model is loaded.
7. Width 390 px: the four steps stack, no horizontal scroll; dark mode is readable.

- [ ] **Step 9: Commit**

```bash
git add site/index.html site/style.css site/js/runner.js site/js/main.js site/tests/runner.test.mjs
git commit -m "Demo site: storyboard page, condition toggle, schematic, live inference and difference map"
```

---

### Task 8: Parity page against PyTorch

**Files:**
- Create: `site/verify.html`, `site/js/verify.js`

**Interfaces:**
- Consumes: Task 6 and 7 modules; `manifest.json` (`samples[].pytorch.on.p`, `samples[].pytorch.off.p`).
- Produces: one row per sample and condition, and `#summary` = `PASS (122 runs, backend …)` or `FAIL (n of 122 mismatches)`.

- [ ] **Step 1: Write `verify.html`**

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ReferNet Parity Check</title><link rel="stylesheet" href="style.css">
<script src="https://cdn.jsdelivr.net/npm/pako@1.0.11/dist/pako.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/upng-js@2.1.0/UPNG.js"></script>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js"></script>
<script type="module" src="js/verify.js"></script></head>
<body><main><h1>Browser vs PyTorch</h1><p id="summary">Running…</p>
<table id="rows" style="font-size:12px;font-family:ui-monospace,monospace">
<tr><th>sample</th><th>arm</th><th>browser p</th><th>PyTorch p</th><th>|Δ|</th><th>decision</th></tr></table></main></body></html>
```

- [ ] **Step 2: Write `verify.js`**

```js
// site/js/verify.js
import { buildQuery, buildOverlap, rgbToNormalizedCHW, decodePng } from "./tensors.js";
import { sigmoid, decide } from "./decision.js";
import { pickProviders, loadSession, runOnce } from "./runner.js";

const W = 800, H = 512;
const png = async (u) => decodePng(await (await fetch(u)).arrayBuffer(), window.UPNG);

async function main() {
  window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const m = await (await fetch("manifest.json")).json();
  const T = window.ort.Tensor, prov = new URLSearchParams(location.search).get("wasm") ? ["wasm"] : pickProviders(navigator);
  const sessions = {};
  let backend = "";
  for (const arm of ["on", "off"]) {
    const s = await loadSession(window.ort, `model/refernet_${arm}.onnx`, prov, () => {});
    sessions[arm] = s.session; backend = s.backend;
  }
  let bad = 0, n = 0;
  for (const s of m.samples) {
    const [q, mk, r] = await Promise.all([png(s.query), png(s.masks), png(s.reference)]);
    const qt = buildQuery(q.rgba, mk.rgba, W, H), ov = buildOverlap(mk.rgba, W, H), rt = rgbToNormalizedCHW(r.rgba, W, H);
    for (const arm of ["on", "off"]) {
      const feeds = { query: new T("float32", qt, [1, 4, H, W]) };
      if (arm === "on") { feeds.reference = new T("float32", rt, [1, 3, H, W]); feeds.overlap = new T("float32", ov, [1, 1, H, W]); }
      const { outputs } = await runOnce(sessions[arm], feeds);
      const p = sigmoid(outputs.query_logit.data[0] + (arm === "on" ? outputs.correction.data[0] : 0));
      const ref = s.pytorch[arm].p, d = Math.abs(p - ref), ok = d <= 1e-3 && decide(p) === decide(ref);
      bad += ok ? 0 : 1; n++;
      document.getElementById("rows").insertAdjacentHTML("beforeend",
        `<tr style="color:${ok ? "inherit" : "#c8102e"}"><td>${s.id}</td><td>${arm}</td><td>${p.toFixed(5)}</td><td>${ref.toFixed(5)}</td><td>${d.toExponential(1)}</td><td>${decide(p)}</td></tr>`);
    }
  }
  document.getElementById("summary").textContent = bad ? `FAIL (${bad} of ${n} mismatches)` : `PASS (${n} runs, backend ${backend})`;
}

main();
```

- [ ] **Step 3: Run it**

Run: `cd site && python3 -m http.server 8765`; open `http://localhost:8765/verify.html` in Chrome, then `http://localhost:8765/verify.html?wasm=1`.
Expected: `PASS (122 runs, backend webgpu)` and `PASS (122 runs, backend wasm)`. If only WebGPU fails, change `pickProviders` to return `["wasm"]` first, rerun Task 7's node tests after updating their expectation, and require PASS on both URLs before continuing.

- [ ] **Step 4: Commit**

```bash
git add site/verify.html site/js/verify.js
git commit -m "Demo site: parity page, browser scores against PyTorch for every sample"
```

---

### Task 9: GitHub Pages deployment

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `README.md` (add a "Demo" section)

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/pages.yml
name: Deploy demo site
on:
  push:
    branches: [main]
    paths: ["site/**", ".github/workflows/pages.yml"]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: node --test site/tests/
  deploy:
    needs: test
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Add the README section**

```markdown
## Demo

A live demo runs the paper's fold-0 deployment model in the browser: https://7abushahla.github.io/ReferNet/ (source in
`site/`). Its data are rebuilt with `python demo/export_onnx.py`, `python demo/dump_inputs.py` (cluster, GPU) and
`python demo/build_site_data.py`; `/verify.html` checks every browser score against PyTorch.
```

- [ ] **Step 3: Validate the workflow file and commit**

```bash
python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/pages.yml')); print('workflow parses')"
git add .github/workflows/pages.yml README.md
git commit -m "Demo: GitHub Pages workflow and README section"
```

- [ ] **Step 4: Enable Pages (only after the repository is renamed to ReferNet, made public, and the user says to push)**

```bash
git push origin main
gh api -X POST repos/7abushahla/ReferNet/pages -f build_type=workflow
gh run watch
```

Expected: `test` and `deploy` succeed; `https://7abushahla.github.io/ReferNet/` loads and `/verify.html` reports PASS.
