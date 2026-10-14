"""Fixed-threshold accuracy of the deployment model on genuine versus in-place text replacement, per IDNet design.

The IDNet text-replacement detector (Guan et al., arXiv 2024) reports accuracy on genuine against text-replaced
documents, trained on one design and tested on another. This script gives the comparable number for the deployment
model (fold 0, OFF and ON) on the Protocol C populations: the sampled genuine renders and their text-replacement copies,
compression equalized, decided at logit 0 (refer when attack logit > 0, ties genuine). Also reports APCER and BPCER
of that decision and the AUC of the pair.

Usage: python scripts/protocol_c_text_replace_accuracy.py  ->  results/tables/v2-sweep/protocol_c_text_replace_accuracy.json
"""
import csv, json
from pathlib import Path

import numpy as np

from protocol_c_full_metrics import auc

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "results/diagnostics/protocol-c-idnet"
C_DESIGNS = {"FIN": "FIN", "GRC": "GRC", "ESP": "ESP", "ALB": "ALB", "AZ": "fold0", "SD": "SD", "WV": "WV", "NV": "NV"}


def main():
    out = {}
    for key, d in C_DESIGNS.items():
        for arm in ("off", "on"):
            rows = [r for r in csv.DictReader(open(C_DIR / d / f"fold0_{arm}_q75_predictions.csv")) if r["family"] in ("genuine", "text_replace")]
            y = np.array([int(r["label"]) for r in rows]); s = np.array([float(r["attack_logit"]) for r in rows]); p = (s > 0).astype(int)
            g, a = y == 0, y == 1
            out[f"{key}:{arm}:q75"] = dict(n_genuine=int(g.sum()), n_replace=int(a.sum()), acc_fixed=float((p == y).mean()),
                                           apcer_fixed=float((p[a] == 0).mean()), bpcer_fixed=float(p[g].mean()), auc=auc(s, y))
    path = ROOT / "results/tables/v2-sweep/protocol_c_text_replace_accuracy.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:14s} n={v['n_genuine']}+{v['n_replace']} acc={100*v['acc_fixed']:.1f} apcer={100*v['apcer_fixed']:.1f} bpcer={100*v['bpcer_fixed']:.1f} auc={v['auc']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
