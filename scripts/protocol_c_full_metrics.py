"""Matched-population metrics for a full-release Protocol C scoring run (every genuine image outside the enrolled
support and every text-family image of identities outside the support, scored by the deployment model).

For each condition (off, on) and file setting (native = as released, q75 = compression equalized) two populations are
summarized: `all` (genuine + the three text families) and `sidtd_families` (genuine + inpaint-and-rewrite +
crop-and-replace), the population of the IDNet paper's SIDTD-trained CNN rows.

Metrics: AUC (genuine vs altered, attack logit), accuracy at the fixed 0.5 threshold (logit 0, ties genuine),
balanced accuracy, genuine referred at that threshold, attacks caught at a 10% genuine referral rate.

Usage: python scripts/protocol_c_full_metrics.py --runs results/diagnostics/protocol-c-idnet/GRC-full \
           --out results/tables/v2-sweep/protocol_c_idnet_GRC_full.json
"""
import argparse, csv, json
from pathlib import Path

import numpy as np


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), float)
    s = scores[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    pos = labels == 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(rows):
    y = np.array([int(r["label"]) for r in rows]); s = np.array([float(r["attack_logit"]) for r in rows])
    pred = (s > 0).astype(int)
    gen, alt = y == 0, y == 1
    thr = np.quantile(s[gen], 0.9)  # refers 10% of genuine
    return dict(n=int(len(rows)), n_genuine=int(gen.sum()), auc=auc(s, y), acc_fixed=float((pred == y).mean()),
                balanced_acc=float(((pred[gen] == 0).mean() + (pred[alt] == 1).mean()) / 2),
                genuine_referred=int(pred[gen].sum()), tpr_fpr10=float((s[alt] > thr).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = {}
    for arm in ("off", "on"):
        for files in ("native", "q75"):
            rows = list(csv.DictReader(open(Path(a.runs) / f"fold0_{arm}_{files}_predictions.csv")))
            out[f"{arm}:{files}:all"] = summarize(rows)
            out[f"{arm}:{files}:sidtd_families"] = summarize([r for r in rows if r["family"] in ("genuine", "text_inpaint", "text_copy")])
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps({k: {m: round(v[m], 3) for m in ("auc", "acc_fixed", "balanced_acc", "tpr_fpr10")} | dict(n=v["n"], ref=v["genuine_referred"]) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
