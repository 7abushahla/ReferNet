"""Read-only diagnostic for the manuscript rewrite (14 Oct 2026).

Protocol B: per held-out design, averaged over the groups that hold it out (each group has its own checkpoint),
AUC overall / inpaint / copy, genuine referred at the fixed 0.5 rule, and mean query and correction logits by class.
Protocol C: deployment model (fold 0), per IDNet design, the same decomposition on compression-equalized files.
"""
import csv
import glob
import os
import statistics as st
import sys

import numpy as np

ROOT = "/shared/b00088568/github/refer-or-clear/results/diagnostics"


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fam_of(r):
    f = (r.get("attack_family") or r.get("family") or "").strip()
    return "genuine" if r["label"] == "0" else (f or "attack")


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def protocol_b(variant):
    per = {}  # (arm, design) -> list of dicts, one per group
    for gdir in sorted(glob.glob(f"{ROOT}/locked-predictions/protocol-b/group*-v2noaux-fdb7351")):
        for arm in ("off", "on"):
            p = f"{gdir}/{variant}/{arm}/test_predictions.csv"
            if not os.path.exists(p):
                continue
            rows = read(p)
            for d in sorted({r["design_id"] for r in rows}):
                rs = [r for r in rows if r["design_id"] == d]
                gen = [float(r["attack_logit"]) for r in rs if r["label"] == "0"]
                fam = {}
                for r in rs:
                    fam.setdefault(fam_of(r), []).append(r)
                att = [float(r["attack_logit"]) for r in rs if r["label"] == "1"]
                rec = dict(auc=auc(att, gen),
                           inpaint=auc([float(r["attack_logit"]) for r in fam.get("text_inpaint", [])], gen),
                           copy=auc([float(r["attack_logit"]) for r in fam.get("text_copy", [])], gen),
                           gen_ref=float(np.mean(np.asarray(gen) > 0)),
                           n_gen=len(gen))
                t10 = float(np.quantile(gen, 0.9))
                rec["caught10"] = float(np.mean(np.asarray(att) > t10))
                rec["copy_caught10"] = float(np.mean(np.asarray([float(r["attack_logit"]) for r in fam.get("text_copy", [])]) > t10))
                for k in ("genuine", "text_inpaint", "text_copy"):
                    rows_k = fam.get(k, [])
                    rec[f"q_{k}"] = mean([float(r["query_logit"]) for r in rows_k])
                    rec[f"c_{k}"] = mean([float(r["correction_logit"]) for r in rows_k])
                per.setdefault((arm, d), []).append(rec)
    print(f"\n=== Protocol B, {variant}: per held-out design, mean over the groups that hold it out")
    print("arm design groups | AUC inpaint copy | genRef@0.5 caught@10% copyCaught@10% | q:gen q:inp q:copy | c:gen c:inp c:copy")
    for (arm, d), recs in sorted(per.items()):
        m = {k: mean([r[k] for r in recs]) for k in recs[0]}
        print(f"{arm:3s} {d:28s} {len(recs)} | {m['auc']:.3f} {m['inpaint']:.3f} {m['copy']:.3f} | "
              f"{m['gen_ref']:.2f} {m['caught10']:.2f} {m['copy_caught10']:.2f} | "
              f"{m['q_genuine']:+.1f} {m['q_text_inpaint']:+.1f} {m['q_text_copy']:+.1f} | "
              f"{m['c_genuine']:+.1f} {m['c_text_inpaint']:+.1f} {m['c_text_copy']:+.1f}")


def protocol_c():
    print("\n=== Protocol C, deployment model (fold 0), equalized files: per design and class")
    print("design arm | n_gen genRef@0.5 | family: n meanQuery meanCorr meanFused | AUC(query) AUC(corr) AUC(fused) vs genuine")
    paths = {"AZ": f"{ROOT}/protocol-c-idnet/fold0"}
    for d in ("ALB", "ESP", "FIN", "GRC", "NV", "SD", "WV"):
        paths[d] = f"{ROOT}/protocol-c-idnet/{d}"
    for d, base in paths.items():
        for arm in ("off", "on"):
            p = f"{base}/fold0_{arm}_q75_predictions.csv"
            if not os.path.exists(p):
                print(d, arm, "missing", p)
                continue
            rows = read(p)
            fam = {}
            for r in rows:
                fam.setdefault(fam_of(r), []).append(r)
            g = fam.get("genuine", [])
            gq = [float(r["query_logit"]) for r in g]
            gc = [float(r["correction_logit"]) for r in g]
            gf = [float(r["attack_logit"]) for r in g]
            line = [f"{d:4s} {arm:3s} | {len(g)} {np.mean(np.asarray(gf) > 0):.2f} | genuine q{mean(gq):+.1f} c{mean(gc):+.1f} f{mean(gf):+.1f}"]
            for k in sorted(x for x in fam if x != "genuine"):
                rs = fam[k]
                q = [float(r["query_logit"]) for r in rs]
                c = [float(r["correction_logit"]) for r in rs]
                fz = [float(r["attack_logit"]) for r in rs]
                line.append(f"{k}: q{mean(q):+.1f} c{mean(c):+.1f} f{mean(fz):+.1f} AUCq {auc(q, gq):.2f} AUCc {auc(c, gc):.2f} AUCf {auc(fz, gf):.2f}")
            print(" || ".join(line))
        # genuine correction quantiles for ON
        p = f"{base}/fold0_on_q75_predictions.csv"
        if os.path.exists(p):
            g = [float(r["correction_logit"]) for r in read(p) if r["label"] == "0"]
            print(f"     {d} ON genuine correction quantiles 10/50/90%: " + " ".join(f"{v:+.1f}" for v in np.quantile(g, [.1, .5, .9])))


if __name__ == "__main__":
    for v in ("evaluation-q75", "evaluation"):
        protocol_b(v)
    protocol_c()
    sys.stdout.flush()
