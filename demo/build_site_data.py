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
