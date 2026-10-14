"""Digest the zero-shot IDNet scoring runs (scripts/diagnostics/protocol-c-capture/c_idnet.py) into one JSON.

Input: one or more run directories holding fold{k}_{arm}_{variant}_predictions.csv (variant native | q75).
Output: per fold and per (arm, variant): overall AUC, text-only AUC (genuine vs the three text families), per-family AUC,
TPR of each family at 10 % genuine referral (threshold = 90th percentile of the genuine attack logits of that fold),
genuine false positives and altered misses at the fixed 0.5 threshold; then mean and sd over folds.

Usage: python scripts/protocol_c_idnet_digest.py --runs DIR [DIR ...] --out results/tables/v2-sweep/protocol_c_idnet.json
"""
import argparse, csv, json, re, statistics
from pathlib import Path
import numpy as np

TEXT = ("text_copy", "text_inpaint", "text_replace")
FACE = ("face_replace", "face_morph")


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n): return None
    return float((p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean())


def digest_file(path):
    rows = list(csv.DictReader(open(path)))
    y = [int(r["label"]) for r in rows]; s = [float(r["attack_logit"]) for r in rows]
    gen = [float(r["attack_logit"]) for r in rows if r["label"] == "0"]; thr = float(np.quantile(gen, .9))
    out = dict(n=len(rows), n_genuine=len(gen), auc=auc(y, s),
               genuine_fp=int(sum(v > 0 for v in gen)), altered_misses=int(sum(float(r["attack_logit"]) <= 0 for r in rows if r["label"] == "1")),
               accuracy=float(np.mean([(v > 0) == bool(t) for v, t in zip(s, y)])))
    text = [r for r in rows if r["family"] == "genuine" or r["family"] in TEXT]
    out["auc_text"] = auc([int(r["label"]) for r in text], [float(r["attack_logit"]) for r in text])
    for fam in TEXT + FACE:
        sub = [r for r in rows if r["family"] == fam]
        if not sub: continue
        fs = [float(r["attack_logit"]) for r in sub]
        out[fam] = dict(n=len(sub), auc=auc([0] * len(gen) + [1] * len(sub), gen + fs), tpr_fpr10=float(np.mean(np.array(fs) > thr)),
                        misses=int(sum(v <= 0 for v in fs)))
    text_scores = [float(r["attack_logit"]) for r in rows if r["family"] in TEXT]
    out["tpr_text_fpr10"] = float(np.mean(np.array(text_scores) > thr))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = {}
    for d in a.runs:
        for p in sorted(Path(d).glob("fold*_predictions.csv")):
            m = re.match(r"fold(\d+)_(off|on)_(native|q75)_predictions\.csv", p.name)
            if not m: continue
            folds.setdefault(f"{m.group(2)}:{m.group(3)}", {})[int(m.group(1))] = digest_file(p)
    agg = {}
    for key, per in folds.items():
        ks = sorted(per); first = per[ks[0]]
        def ms(get):
            vals = [get(per[k]) for k in ks if get(per[k]) is not None]
            return dict(mean=float(statistics.mean(vals)), sd=float(statistics.stdev(vals)) if len(vals) > 1 else 0.0, min=min(vals), max=max(vals), n=len(vals))
        agg[key] = dict(folds=ks, auc=ms(lambda s: s["auc"]), auc_text=ms(lambda s: s["auc_text"]), accuracy=ms(lambda s: s["accuracy"]),
                        genuine_fp=ms(lambda s: s["genuine_fp"]), tpr_text_fpr10=ms(lambda s: s["tpr_text_fpr10"]),
                        **{fam: dict(auc=ms(lambda s, fam=fam: s[fam]["auc"]), tpr_fpr10=ms(lambda s, fam=fam: s[fam]["tpr_fpr10"])) for fam in TEXT + FACE if fam in first})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(per_fold=folds, aggregate=agg, n_queries=first["n"], n_genuine=first["n_genuine"]), indent=1, sort_keys=True))
    for key, v in sorted(agg.items()):
        print(f"{key:12s} folds={len(v['folds'])} AUC {v['auc']['mean']:.3f}±{v['auc']['sd']:.3f} text {v['auc_text']['mean']:.3f}±{v['auc_text']['sd']:.3f} "
              + " ".join(f"{fam} {v[fam]['auc']['mean']:.3f}" for fam in TEXT + FACE if fam in v) + f" TPR@10% text {v['tpr_text_fpr10']['mean']:.3f} genuineFP {v['genuine_fp']['mean']:.0f}")


if __name__ == "__main__":
    main()
