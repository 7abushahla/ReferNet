"""ISO/IEC 30107-3 error rates from the locked prediction files of the three protocols.

For every unit (Protocol A fold, Protocol B group, Protocol C design) and every model, from the attack logit
(a document is referred when the logit exceeds the threshold):
  eer               equal error rate: the threshold where APCER and BPCER are closest, reported as their mean there
  apcer_at_bpcer_X  APCER at the threshold that refers X% of the genuine documents (X = 1, 5, 10), i.e. 1 - attacks caught
  apcer_fixed, bpcer_fixed, hter_fixed   at the fixed 0.5 rule (logit 0, ties genuine); HTER is their mean
Units are summarized by mean, sd, min, max over folds (A) or groups (B); Protocol C is per design.

Inputs (all in this repository):
  A   results/diagnostics/locked-predictions/protocol-a/fold<k>-v2noaux-fdb7351/{evaluation,evaluation-q75}/<arm>/test_predictions.csv
  B   results/diagnostics/locked-predictions/protocol-b/group<k>-v2noaux-fdb7351/{evaluation,evaluation-q75}/<arm>/test_predictions.csv
      results/diagnostics/locked-predictions/baselines-b/group<k>/<model>/{native,q75}_predictions.csv
  C   results/diagnostics/protocol-c-idnet/<design>/fold0_<arm>_q75_predictions.csv (genuine + the three text families),
      and the full-release files FIN-full, GRC-full (genuine + the two SIDTD families, as in the published comparison)

Usage: python scripts/iso_error_rates.py --out results/tables/v2-sweep/iso_error_rates.json
"""
import argparse, csv, json, statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "results/diagnostics/locked-predictions"
C_DIR = ROOT / "results/diagnostics/protocol-c-idnet"
C_DESIGNS = {"FIN": "FIN", "GRC": "GRC", "ESP": "ESP", "ALB": "ALB", "AZ": "fold0", "SD": "SD", "WV": "WV", "NV": "NV"}
TEXT = ("genuine", "text_inpaint", "text_copy", "text_replace")
BPCER_TARGETS = (0.01, 0.05, 0.10)


def read(path, families=None):
    rows = list(csv.DictReader(open(path)))
    if families is not None:
        rows = [r for r in rows if r["family"] in families]
    return np.array([float(r["attack_logit"]) for r in rows]), np.array([int(r["label"]) for r in rows])


def rates(s, y):
    gen, att = s[y == 0], s[y == 1]
    out = dict(n_genuine=int(len(gen)), n_altered=int(len(att)),
               bpcer_fixed=float((gen > 0).mean()), apcer_fixed=float((att <= 0).mean()))
    out["hter_fixed"] = (out["bpcer_fixed"] + out["apcer_fixed"]) / 2
    for x in BPCER_TARGETS:
        t = float(np.quantile(gen, 1 - x))  # refers at most X% of genuine
        out[f"apcer_at_bpcer_{int(round(100 * x))}"] = float((att <= t).mean())
        out[f"bpcer_realized_{int(round(100 * x))}"] = float((gen > t).mean())
    # EER over every achievable threshold (midpoints between sorted scores plus the extremes)
    u = np.unique(s); cands = np.concatenate(([u[0] - 1.], (u[:-1] + u[1:]) / 2., [u[-1] + 1.]))
    best = None
    for t in cands:
        b, a = float((gen > t).mean()), float((att <= t).mean())
        key = (abs(a - b), (a + b) / 2)
        if best is None or key < best[0]:
            best = (key, t, a, b)
    out["eer"] = (best[2] + best[3]) / 2; out["eer_threshold"] = float(best[1]); out["eer_apcer"] = best[2]; out["eer_bpcer"] = best[3]
    return out


def summarize(units):
    keys = [k for k in units[0] if k not in ("n_genuine", "n_altered", "eer_threshold")]
    return {k: dict(mean=statistics.mean(v), sd=statistics.stdev(v) if len(v) > 1 else 0., min=min(v), max=max(v), n=len(v))
            for k in keys for v in [[u[k] for u in units]]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = {"A": {}, "B": {}, "C": {}}
    for files, sub in (("native", "evaluation"), ("q75", "evaluation-q75")):
        for arm in ("off", "on"):
            units = [rates(*read(LOCKED / f"protocol-a/fold{k}-v2noaux-fdb7351/{sub}/{arm}/test_predictions.csv")) for k in range(10)]
            out["A"][f"{arm}:{files}"] = dict(units=units, summary=summarize(units))
            units = [rates(*read(LOCKED / f"protocol-b/group{k}-v2noaux-fdb7351/{sub}/{arm}/test_predictions.csv")) for k in range(10)]
            out["B"][f"{arm}:{files}"] = dict(units=units, summary=summarize(units))
        for model in ("resnet50", "efficientnet-b3"):
            units = [rates(*read(LOCKED / f"baselines-b/group{k}/{model}/{files}_predictions.csv")) for k in range(10)]
            out["B"][f"{model}:{files}"] = dict(units=units, summary=summarize(units))
    for key, d in C_DESIGNS.items():
        for arm in ("off", "on"):
            out["C"][f"{key}:{arm}:q75:text"] = rates(*read(C_DIR / d / f"fold0_{arm}_q75_predictions.csv", TEXT))
    for key in ("FIN", "GRC"):
        for arm in ("off", "on"):
            out["C"][f"{key}-full:{arm}:q75:sidtd_families"] = rates(*read(C_DIR / f"{key}-full" / f"fold0_{arm}_q75_predictions.csv", ("genuine", "text_inpaint", "text_copy")))
    Path(a.out).write_text(json.dumps(out, indent=1))
    fmt = lambda r: " ".join(f"{k}={r[k]:.3f}" for k in ("eer", "apcer_at_bpcer_1", "apcer_at_bpcer_5", "apcer_at_bpcer_10", "apcer_fixed", "bpcer_fixed", "hter_fixed"))
    for p in ("A", "B"):
        for k, v in out[p].items():
            print(p, k, fmt({m: v["summary"][m]["mean"] for m in v["summary"]}))
    for k, v in out["C"].items():
        print("C", k, fmt(v))


if __name__ == "__main__":
    main()
