"""Write docs/supplement.md: the full result records behind the curated tables of the award manuscript.

The manuscript carries one curated table per protocol; everything else measured on the frozen checkpoints is listed here
from the same JSON records (results/tables/v2-sweep/*.json, results/baselines/protocol-b/*/summary.json), so that every
number can be traced to a prediction file.

Usage: python scripts/render_supplement.py [--out docs/supplement.md]
"""
import argparse, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results/tables/v2-sweep"
DESIGNS = [("FIN", "Finland identity card", 1), ("GRC", "Greece passport", 1), ("ESP", "Spain identity card", 1), ("ALB", "Albania identity card", 1),
           ("AZ", "Arizona licence", 2), ("SD", "South Dakota licence", 2), ("WV", "West Virginia licence", 2), ("NV", "Nevada licence", 2)]
CUE_KEPT = {"native": (91.9, 0.982, 0.969, 0.984), "q75": (57.7, 0.548, 0.517, 0.554)}  # OFF network trained on raw fold-0 files at 320x320


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def ms(v, nd=3):
    return f"{statistics.mean(v):.{nd}f} ± {statistics.stdev(v):.{nd}f}" if len(v) > 1 else "--"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def protocol_a(d):
    out = ["## S1. Protocol A: per-fold records\n",
           "Official SIDTD template folds, test partitions. Selection on each fold's validation partition (accuracy, then AUC, then earlier epoch).",
           "`as released` scores the test files as distributed; `equalized` re-saves every test file at IJG JPEG quality 75 before any resizing, so that the genuine (quality 92) and altered (quality 75) files share one quantization table.\n"]
    for files, key in (("as released", "test"), ("equalized", "q75")):
        rows = []
        for u in d["A"]:
            k = u["unit"]
            for arm in ("off", "on"):
                v = u[f"{key}_{arm}"]
                rows.append([k, arm.upper(), v["epoch"], f"{100 * v['acc']:.1f}", f3(v["auc"]), f3(v["copy_auc"]), f3(v["inpainting_auc"]), v["fp"], v["misses"], v["n_unavailable"]])
        for arm in ("off", "on"):
            vs = [u[f"{key}_{arm}"] for u in d["A"]]
            rows.append(["mean ± sd", arm.upper(), "", ms([100 * v["acc"] for v in vs], 1), ms([v["auc"] for v in vs]), ms([v["copy_auc"] for v in vs]), ms([v["inpainting_auc"] for v in vs]),
                         sum(v["fp"] for v in vs), sum(v["misses"] for v in vs), sum(v["n_unavailable"] for v in vs)])
        out.append(f"### Files {files}\n")
        out.append(table(["fold", "condition", "epoch", "accuracy (%)", "AUC", "copy-and-replace AUC", "inpaint-and-rewrite AUC", "genuine referred", "attacks missed", "context unavailable"], rows))
    out.append("### Control: our single-image network trained on the raw files\n")
    out.append("OFF network trained on the raw fold-0 files resized to 320×320 (the published SIDTD input path, so the quantization cue reaches it), scored on the fold-0 test partition.\n")
    out.append(table(["files", "accuracy (%)", "AUC", "copy-and-replace AUC", "inpaint-and-rewrite AUC"],
                     [["as released", *CUE_KEPT["native"]], ["equalized", *CUE_KEPT["q75"]]]))
    return "\n".join(out)


def protocol_b(ops):
    out = ["## S2. Protocol B: per-group records, including the development group\n",
           "Group 0 is the development group on which the ON recipe was fixed (see the manuscript, recipe contrast); it is excluded from every mean in the manuscript and listed here for completeness. "
           "`fixed` is the 0.5 alteration-probability threshold (logit 0); `calibrated` is the logit threshold with the highest accuracy on the group's own source validation partition; "
           "`caught at 10%` is the fraction of attacks above the threshold that refers 10% of the group's genuine queries.\n"]
    units = sorted(ops["units"], key=lambda u: u["group"])
    rows = [[u["group"], u["group_name"], u["published"], u["support"], u["excluded"], u["queries"], u["off_test"]["n_genuine"], u["off_test"]["n_altered"],
             u["off_test"]["text_inpaint"]["n"], u["off_test"]["text_copy"]["n"]] for u in units]
    out.append("### Populations\n")
    out.append(table(["group", "held-out nationalities", "published images", "enrollment support", "excluded (support identities and linked attacks)", "queries", "genuine", "altered", "inpaint-and-rewrite", "copy-and-replace"], rows))
    for files, key in (("as released", "test"), ("equalized", "q75")):
        rows = []
        for u in units:
            for arm in ("off", "on"):
                v = u[f"{arm}_{key}"]
                rows.append([u["group"], arm.upper(), f3(v["auc"]), f3(v["text_inpaint"]["auc"]), f3(v["text_copy"]["auc"]), f"{100 * v['acc_fixed']:.1f}", f"{v['referred_fixed']}/{v['n_genuine']}", v["missed_fixed"],
                             f"{v['threshold_cal']:.2f}", f"{100 * v['acc_cal']:.1f}", f"{v['referred_cal']}/{v['n_genuine']}", f3(v["tpr_fpr10"]), f3(v["text_inpaint"]["tpr_fpr10"]), f3(v["text_copy"]["tpr_fpr10"])])
        out.append(f"### OFF and ON, files {files}\n")
        out.append(table(["group", "condition", "AUC", "inpaint AUC", "copy AUC", "accuracy, fixed (%)", "genuine referred, fixed", "attacks missed, fixed", "calibrated logit threshold", "accuracy, calibrated (%)", "genuine referred, calibrated", "caught at 10%", "inpaint caught", "copy caught"], rows))
    rows = []
    for u in units:
        k = u["group"]
        for m in ("resnet50", "efficientnet-b3"):
            p = ROOT / f"results/baselines/protocol-b/group{k}/{m}/summary.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            for files, key in (("as released", "native"), ("equalized", "q75")):
                v = s[key]
                rows.append([k, m, s["selected_epoch"], f"{100 * s['val_acc']:.1f}", files, f3(v["auc"]), f3(v["text_inpaint"]["auc"]), f3(v["text_copy"]["auc"]), f"{100 * v['accuracy']:.1f}", f"{v['genuine_fp']}/{v['n_genuine']}", v["altered_misses"], f3(v["tpr_fpr10"])])
    out.append("### Published single-image baselines retrained per group with the SIDTD recipe\n")
    out.append("ImageNet-initialized ResNet50 and EfficientNet-B3, 299×299 input, random resized crops, flips and brightness jitter, SGD 0.01 with momentum 0.9, batch 32, 100 epochs, checkpoint by validation accuracy; trained on the six source nationalities of each group, scored on the same frozen queries as OFF and ON (scripts/baselines/sidtd_baseline_protocol_b.py).\n")
    out.append(table(["group", "model", "selected epoch", "source validation accuracy (%)", "files", "AUC", "inpaint AUC", "copy AUC", "accuracy at 0.5 (%)", "genuine referred", "attacks missed", "caught at 10%"], rows))
    return "\n".join(out)


def protocol_c():
    out = ["## S3. Protocol C: IDNet designs, both file settings, all ten split iterations\n",
           "Production model = the fold-0 checkpoints (SIDTD published split iteration 0). `as released` scores the IDNet files as distributed (genuine PNG; inpaint-and-rewrite and crop-and-replace JPEG; in-place replacement PNG); "
           "`equalized` re-saves every query at JPEG quality 75. 100 enrolled genuine images per design; 1,200 queries (300 genuine, 300 per text family) of identities outside the support. "
           "`genuine referred` counts genuine queries above the fixed 0.5 threshold; `caught at 10%` is the fraction of text attacks above the threshold that refers 10% of the genuine queries.\n"]
    for key, label, level in DESIGNS:
        p = TAB / ("protocol_c_idnet.json" if key == "AZ" else f"protocol_c_idnet_{key}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text()); out.append(f"### {label} (level {level})\n")
        rows = []
        for files, fk in (("as released", "native"), ("equalized", "q75")):
            for arm in ("off", "on"):
                for fold in sorted(d["per_fold"][f"{arm}:{fk}"], key=int):
                    v = d["per_fold"][f"{arm}:{fk}"][fold]
                    rows.append([files, arm.upper(), fold, f3(v["auc_text"]), f3(v["text_inpaint"]["auc"]), f3(v["text_copy"]["auc"]), f3(v["text_replace"]["auc"]), f3(v["tpr_text_fpr10"]), f"{v['genuine_fp']}/{v['n_genuine']}", f"{100 * v['accuracy']:.1f}"])
                a = d["aggregate"][f"{arm}:{fk}"]
                rows.append([files, arm.upper(), "mean ± sd", f"{a['auc_text']['mean']:.3f} ± {a['auc_text']['sd']:.3f}", "", "", "", "", "", ""])
        out.append(table(["files", "condition", "split iteration", "text AUC", "inpaint AUC", "copy AUC", "in-place replacement AUC", "caught at 10%", "genuine referred", "accuracy at 0.5 (%)"], rows))
    return "\n".join(out)


def protocol_c_full():
    p = TAB / "protocol_c_idnet_FIN_full.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    out = ["## S4. Protocol C: Finland identity card, full release (production model)\n",
           "Every released genuine image outside the 100 enrolled (5,879) and every text-family image of identities outside the support, scored by the fold-0 checkpoints. "
           "`all` = genuine + inpaint-and-rewrite + crop-and-replace + in-place replacement (23,346 documents); `sidtd_families` = genuine + the two SIDTD-style families (17,497), the population of IDNet's Tables II/III (SIDTD-trained CNNs: 67.2%).\n"]
    rows = [[k.split(":")[0].upper(), k.split(":")[1].replace("native", "as released").replace("q75", "equalized"), k.split(":")[2], v["n"], v["n_genuine"], f3(v["auc"]), f"{100 * v['acc_fixed']:.1f}", f"{100 * v['balanced_acc']:.1f}", f"{v['genuine_referred']}/{v['n_genuine']}", f3(v["tpr_fpr10"])] for k, v in d.items()]
    out.append(table(["condition", "files", "population", "n", "genuine", "AUC", "accuracy at 0.5 (%)", "balanced accuracy (%)", "genuine referred", "caught at 10%"], rows))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "docs/supplement.md")); a = ap.parse_args()
    d = json.loads((TAB / "digest.json").read_text()); ops = json.loads((TAB / "protocol_b_operating_points.json").read_text())
    head = ["# Supplementary results for \"Refer or Clear\" (GDRFA award manuscript)\n",
            "Generated by `scripts/render_supplement.py` from the frozen result records in this repository. The manuscript reports the curated tables; this file lists every measurement behind them: "
            "per-fold Protocol A records and the compression control, per-group Protocol B records including the development group, the retrained published baselines, and Protocol C on both file settings for all ten split iterations.\n",
            "Sources: `results/tables/v2-sweep/digest.json` (A and B), `results/tables/v2-sweep/protocol_b_operating_points.json` (B operating points), `results/baselines/protocol-b/group*/*/summary.json` (baselines), "
            "`results/tables/v2-sweep/protocol_c_idnet*.json` (C), with the prediction files under `results/runs/` and `results/diagnostics/protocol-c-idnet/`.\n"]
    Path(a.out).write_text("\n".join(head + [protocol_a(d), protocol_b(ops), protocol_c(), protocol_c_full()])); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
