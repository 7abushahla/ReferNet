#!/usr/bin/env python3
"""Aggregate results/runs/<tag>/metrics.json into per-arm tables (mean and std over seeds).

    python scripts/aggregate_runs.py [--runs results/runs] [--protocol A] [--out results/tables/protocol_A.md]

Tag format: <protocol>-<arch>-<fusion>-<reference>-<init>-s<seed>; everything but the seed is the arm.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

METRICS = ["auc", "eer", "deepid_acc", "deepid_f1", "transported_miss5", "transported_referral5",
           "transported_miss1", "transported_referral1", "bpcer_at_apcer_5", "loc_f1", "fp_area_in_zone", "fp_area_out_zone"]
ARM_LABEL = {("single", "median"): "standard, control", ("difference", "median"): "standard, median",
             ("difference", "specimen"): "standard, specimen", ("attention_single", "median"): "improved, control",
             ("attention", "median"): "improved, median", ("attention", "specimen"): "improved, specimen"}
TAG = re.compile(r"^(?P<protocol>[A-D])-(?P<arch>[a-z0-9]+)-(?P<fusion>[a-z_]+)-(?P<reference>[a-z]+)-(?P<init>[a-z]+)"
                 r"(?:-f(?P<fold>\d+))?-s(?P<seed>\d+)$")
FOLDED = {"A", "B"}   # protocols whose runs carry a fold or group index; foldless runs there are superseded
OURS = {("single", "median"): "Ours, control (no reference)", ("difference", "median"): "Ours, design median",
        ("difference", "specimen"): "Ours, blank specimen"}
# Boned et al., Scientific Data 11:1356 (2024), Table 5, Composite Detection (K-fold, Balanced):
# (model, accuracy %, sd, ROC AUC, sd), as published.
PUBLISHED_A = [("EfficientNet-B3", 99.4, 1.0, 1.00, 0.00), ("ResNet50", 98.1, 1.2, 0.99, 0.01),
               ("ViT-L/16", 55.2, 2.3, 0.50, 0.04), ("TransFG", 96.6, 1.5, 0.99, 0.01),
               ("CoAARN", 98.6, 1.6, 0.99, 0.01)]
GATE = dict(auc=0.99, acc=0.97, n=10)


def load(runs: Path, protocol: str) -> pd.DataFrame:
    rows = []
    for mj in sorted(runs.glob("*/metrics.json")):
        m = TAG.match(mj.parent.name)
        if not m or m["protocol"] != protocol:
            continue
        if protocol in FOLDED and m["fold"] is None:
            continue
        d = json.load(open(mj))
        for variant, block in d["test"].items():
            rows.append(dict(**m.groupdict(), variant=variant, val_auc=d["val"].get("auc"),
                             epochs=d["timing"].get("epochs"), best_epoch=d["timing"].get("best_epoch"),
                             **{k: block.get(k) for k in METRICS}))
        for subset, blocks in d.get("subsets", {}).items():
            for variant, block in blocks.items():
                rows.append(dict(**m.groupdict(), variant=f"{variant}:{subset}", val_auc=None, epochs=None, best_epoch=None,
                                 **{k: block.get(k) for k in METRICS}))
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def table(df: pd.DataFrame, variant: str, cols: list[str]) -> str:
    sub = df[df.variant == variant].copy()
    if sub.empty:
        return f"(no runs for variant {variant})"
    sub["arm"] = [ARM_LABEL.get((f, r), f"{f}, {r}") for f, r in zip(sub.fusion, sub.reference)]
    g = sub.groupby("arm")
    out = ["| arm | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    order = list(ARM_LABEL.values())
    for arm in sorted(g.groups, key=lambda a: order.index(a) if a in order else 99):
        s = g.get_group(arm)
        cells = []
        for c in cols:
            v = pd.to_numeric(s[c], errors="coerce")
            cells.append("" if v.isna().all() else (f"{v.mean():.3f} ± {v.std(ddof=0):.3f}" if len(v) > 1 else f"{v.mean():.3f}"))
        out.append(f"| {arm} | {len(s)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _ms(v: pd.Series, scale: float = 1.0, nd: int = 3) -> str:
    v = pd.to_numeric(v, errors="coerce").dropna() * scale
    if v.empty:
        return ""
    return f"{v.mean():.{nd}f} ± {v.std(ddof=0):.{nd}f}"


def paper_table(df: pd.DataFrame, variant: str) -> str:
    """Table 5 layout: the published rows (Protocol A only), then our arms; accuracy in percent."""
    out = ["| Model | n | Accuracy (%) | ROC AUC |", "|---|---|---|---|"]
    if df.protocol.iloc[0] == "A" and variant == "native":
        out += [f"| {m} (Boned et al. 2024) | 10 | {a:.1f} ± {sa:.1f} | {u:.2f} ± {su:.2f} |" for m, a, sa, u, su in PUBLISHED_A]
    sub = df[df.variant == variant]
    for (f, r), label in OURS.items():
        s = sub[(sub.fusion == f) & (sub.reference == r)]
        if len(s):
            out.append(f"| {label} | {len(s)} | {_ms(s.deepid_acc, 100, 1)} | {_ms(s.auc)} |")
    return "\n".join(out)


def per_design(df: pd.DataFrame) -> str:
    """Per test nationality: mean over the groups in which it was tested (native variant)."""
    sub = df[df.variant.str.startswith("native:design:")].copy()
    if sub.empty:
        return "(no per-design subsets)"
    sub["design"] = sub.variant.str.replace("native:design:", "", regex=False)
    out = ["| Arm | Design | groups | Accuracy (%) | ROC AUC |", "|---|---|---|---|---|"]
    for (f, r), label in OURS.items():
        s = sub[(sub.fusion == f) & (sub.reference == r)]
        for d, g in s.groupby("design"):
            out.append(f"| {label} | {d} | {len(g)} | {_ms(g.deepid_acc, 100, 1)} | {_ms(g.auc)} |")
    return "\n".join(out)


def gate(df: pd.DataFrame) -> tuple[bool, str]:
    """The control on the raw test files: all folds present, mean AUC >= 0.99 and mean accuracy >= 0.97."""
    s = df[(df.variant == "native") & (df.fusion == "single") & (df.reference == "median")]
    n, auc_m, acc_m = len(s), pd.to_numeric(s.auc).mean(), pd.to_numeric(s.deepid_acc).mean()
    if n < GATE["n"]:
        return False, f"GATE FAIL: control has {n} of {GATE['n']} folds"
    fails = []
    if not auc_m >= GATE["auc"]:
        fails.append(f"AUC {auc_m:.4f} < {GATE['auc']}")
    if not acc_m >= GATE["acc"]:
        fails.append(f"accuracy {acc_m:.4f} < {GATE['acc']}")
    if fails:
        return False, "GATE FAIL: " + "; ".join(fails)
    return True, f"GATE PASS: control AUC {auc_m:.4f}, accuracy {acc_m:.4f} over {n} folds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate", action="store_true", help="print the Protocol A gate and exit 1 on FAIL")
    a = ap.parse_args()
    df = load(Path(a.runs), a.protocol)
    if df.empty:
        raise SystemExit(f"no metrics.json under {a.runs} for protocol {a.protocol}")
    cols = ["val_auc", "auc", "eer", "deepid_f1", "transported_miss5", "transported_referral5", "transported_miss1",
            "transported_referral1", "loc_f1", "fp_area_in_zone", "fp_area_out_zone", "best_epoch"]
    parts = [f"# Protocol {a.protocol}: {df[df.variant == 'native'].shape[0]} runs\n"]
    parts += ["\n## Table 5 layout, raw test files\n", paper_table(df, "native"),
              "\n\n## Table 5 layout, compression-equalized test\n", paper_table(df, "equalized_qf75")]
    if a.protocol == "B":
        parts += ["\n\n## Per test nationality\n", per_design(df)]
    for variant in sorted(df.variant.unique()):
        parts += [f"\n## {variant}\n", table(df, variant, cols)]
    per = df[df.variant == "native"][["fusion", "reference", "fold", "seed", "val_auc", "auc", "transported_miss5",
                                      "transported_referral5", "loc_f1", "best_epoch"]]
    parts += ["\n\n## Per run (native)\n", md(per)]
    text = "\n".join(parts) + "\n"
    out = Path(a.out or f"results/tables/protocol_{a.protocol}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if a.gate:
        ok, msg = gate(df)
        print(msg)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
