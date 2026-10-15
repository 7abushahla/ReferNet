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
