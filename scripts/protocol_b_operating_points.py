"""Protocol B operating points from the locked prediction files, without touching the held-out queries for selection.

For each study root <root> (one per group) and each arm:
  threshold_cal   chosen on the SOURCE validation predictions (training/<arm>/val_predictions.csv): the logit threshold
                  with the highest validation accuracy (ties: the one closest to 0, i.e. closest to the fixed 0.5 rule)
  test metrics    on evaluation/<arm>/test_predictions.csv (and evaluation-q75/<arm>/ when present):
                  AUC from the logit; accuracy and genuine referrals at the fixed 0.5 rule (logit 0) and at threshold_cal;
                  TPR at a 10% genuine referral rate (threshold = 90th percentile of the genuine logits), overall and per family.
Families come from the request's records file, because the locked prediction CSVs leave attack_family empty.

Usage: python scripts/protocol_b_operating_points.py --runs "results/runs/protocol-b-current/group*-v2noaux-fdb7351" \
           --requests results/runs/protocol-b-current/requests/v2noaux-fdb7351 --out results/tables/v2-sweep/protocol_b_operating_points.json
"""
import argparse, csv, glob, json, re, statistics
from pathlib import Path

import numpy as np

DEV = 0
FPR = 0.10


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int); p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return None
    return float((p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean())


def read(path):
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r["attack_logit"]) for r in rows]), np.array([int(r["label"]) for r in rows]), [r["image_id"] for r in rows]


def calibrate(logit, label):
    """Logit threshold with the highest validation accuracy; ties resolved toward 0 (the fixed rule). Prediction is attack when logit > t."""
    s = np.sort(np.unique(logit))
    cands = np.concatenate(([s[0] - 1.], (s[:-1] + s[1:]) / 2., [s[-1] + 1.], [0.]))
    best = None
    for t in cands:
        acc = float(np.mean((logit > t) == (label == 1)))
        key = (acc, -abs(t))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1], best[0][0]


def metrics(logit, label, fam, t_cal):
    gen, att = logit[label == 0], logit[label == 1]
    t10 = float(np.quantile(gen, 1 - FPR)) if len(gen) else None  # attack if logit > t10 -> at most 10% of genuine referred
    out = dict(n=int(len(logit)), n_genuine=int(len(gen)), n_altered=int(len(att)), auc=auc(logit, label),
               acc_fixed=float(np.mean((logit > 0) == (label == 1))), referred_fixed=int((gen > 0).sum()), missed_fixed=int((att <= 0).sum()),
               threshold_cal=t_cal, acc_cal=float(np.mean((logit > t_cal) == (label == 1))), referred_cal=int((gen > t_cal).sum()), missed_cal=int((att <= t_cal).sum()),
               threshold_fpr10=t10, tpr_fpr10=float(np.mean(att > t10)) if t10 is not None and len(att) else None,
               referred_fpr10=int((gen > t10).sum()) if t10 is not None else None)
    for f in sorted(set(fam[label == 1])):
        sub = logit[(label == 1) & (fam == f)]
        out[f] = dict(n=int(len(sub)), auc=auc(np.concatenate([gen, sub]), np.array([0] * len(gen) + [1] * len(sub))),
                      recall_fixed=float(np.mean(sub > 0)), recall_cal=float(np.mean(sub > t_cal)),
                      tpr_fpr10=float(np.mean(sub > t10)) if t10 is not None else None)
    return out


def group_of(root):
    m = re.search(r"group(\d+)", Path(root).name)
    return int(m.group(1)) if m else None


def families(request_path):
    req = json.loads(Path(request_path).read_text())
    rec = req["records"]["path"] if isinstance(req["records"], dict) else req["records"]
    return {r["image_id"]: (r["attack_family"] if r["label"] not in ("0", "False", "false") else "genuine") for r in csv.DictReader(open(rec))}


def collect(runs, requests):
    units = []
    for root in sorted(glob.glob(runs), key=group_of):
        root = Path(root); g = group_of(root)
        req = json.loads((Path(requests) / f"group{g}" / "request.json").read_text())
        fam_map = families(Path(requests) / f"group{g}" / "request.json")
        enrol = json.loads(Path(req["target_enrollment"]["path"]).read_text())
        unit = dict(group=g, root=str(root), group_name=enrol.get("group_name"), support=len(enrol["support_ids"]),
                    excluded=len(enrol["excluded_ids"]), queries=len(enrol["query_ids"]), published=enrol.get("published_target_count"))
        for arm in ("off", "on"):
            vp, tp = root / "training" / arm / "val_predictions.csv", root / "evaluation" / arm / "test_predictions.csv"
            if not (vp.exists() and tp.exists()):
                continue
            vl, vy, _ = read(vp); t_cal, val_acc = calibrate(vl, vy)
            unit[f"{arm}_val"] = dict(n=int(len(vl)), acc_fixed=float(np.mean((vl > 0) == (vy == 1))), acc_cal=val_acc, threshold_cal=t_cal)
            for phase, path in (("test", tp), ("q75", root / "evaluation-q75" / arm / "test_predictions.csv")):
                if path.exists():
                    l, y, ids = read(path); fam = np.array([fam_map[i] for i in ids])
                    unit[f"{arm}_{phase}"] = metrics(l, y, fam, t_cal)
        units.append(unit)
    return units


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(n=len(values), mean=float(statistics.mean(values)), sd=float(statistics.stdev(values)) if len(values) > 1 else 0.)


def summarize(units):
    keys = ["auc", "acc_fixed", "acc_cal", "tpr_fpr10", "referred_fixed", "referred_cal"]
    fams = ["text_inpaint", "text_copy"]
    out = {}
    for phase in ("test", "q75"):
        for arm in ("off", "on"):
            held = [u[f"{arm}_{phase}"] for u in units if u["group"] != DEV and f"{arm}_{phase}" in u]
            s = {k: mean_sd([m[k] for m in held]) for k in keys}
            for f in fams:
                s[f"{f}_auc"] = mean_sd([m[f]["auc"] for m in held if f in m])
                s[f"{f}_tpr_fpr10"] = mean_sd([m[f]["tpr_fpr10"] for m in held if f in m])
            s["groups"] = sorted(u["group"] for u in units if u["group"] != DEV and f"{arm}_{phase}" in u)
            out[f"{arm}_{phase}_held_out"] = s
        both = [u for u in units if u["group"] != DEV and f"off_{phase}" in u and f"on_{phase}" in u]
        out[f"gap_{phase}_held_out"] = mean_sd([u[f"on_{phase}"]["auc"] - u[f"off_{phase}"]["auc"] for u in both])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--requests", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    units = collect(a.runs, a.requests)
    result = dict(fpr=FPR, development_group=DEV, calibration="source validation accuracy, ties toward the fixed rule", units=units, summary=summarize(units))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    for u in units:
        for arm in ("off", "on"):
            m = u.get(f"{arm}_test")
            if m:
                print(f"group {u['group']} {arm}: auc {m['auc']:.3f} acc@.5 {m['acc_fixed']:.3f} ref {m['referred_fixed']}/{m['n_genuine']} | "
                      f"t_cal {m['threshold_cal']:+.2f} acc@cal {m['acc_cal']:.3f} ref {m['referred_cal']} | tpr@10%fpr {m['tpr_fpr10']:.3f} "
                      f"(inpaint {m.get('text_inpaint', {}).get('tpr_fpr10', float('nan')):.3f}, copy {m.get('text_copy', {}).get('tpr_fpr10', float('nan')):.3f})")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
