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
TAG = re.compile(r"^(?P<protocol>[ABC])-(?P<arch>[a-z0-9]+)-(?P<fusion>[a-z_]+)-(?P<reference>[a-z]+)-(?P<init>[a-z]+)-s(?P<seed>\d+)$")


def load(runs: Path, protocol: str) -> pd.DataFrame:
    rows = []
    for mj in sorted(runs.glob("*/metrics.json")):
        m = TAG.match(mj.parent.name)
        if not m or m["protocol"] != protocol:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load(Path(a.runs), a.protocol)
    if df.empty:
        raise SystemExit(f"no metrics.json under {a.runs} for protocol {a.protocol}")
    cols = ["val_auc", "auc", "eer", "deepid_f1", "transported_miss5", "transported_referral5", "transported_miss1",
            "transported_referral1", "loc_f1", "fp_area_in_zone", "fp_area_out_zone", "best_epoch"]
    parts = [f"# Protocol {a.protocol}: {df.seed.nunique()} seed(s), {df[df.variant == 'native'].shape[0]} runs\n"]
    for variant in sorted(df.variant.unique()):
        parts += [f"\n## {variant}\n", table(df, variant, cols)]
    parts += ["\n\n## Per run (native)\n", df[df.variant == "native"][["fusion", "reference", "seed", "val_auc", "auc", "transported_miss5", "transported_referral5", "loc_f1", "best_epoch"]].to_markdown(index=False) if hasattr(df, "to_markdown") else ""]
    text = "\n".join(parts) + "\n"
    out = Path(a.out or f"results/tables/protocol_{a.protocol}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
