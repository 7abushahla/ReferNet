"""Aggregate v2 matched OFF/ON study outputs (Protocol A folds, Protocol B groups, C) into tables.

Reads the receipts written by scripts/protocol_a_study.py, scripts/protocol_b_study.py and the C scorer:
  <root>/training/{off,on}/val_summary.json      selected source-validation summary (+selected_epoch)
  <root>/evaluation/{off,on}/test_summary.json   locked target/test summary (if evaluated)
Writes Markdown and LaTeX tables plus a JSON digest. Metadata only; no model execution.

Usage:
  python scripts/aggregate_v2_sweep.py --a "results/runs/protocol-a-current/fold*-v2-dd79473" \
      --b "results/runs/protocol-b-current/group*-v2-dd79473" --c results/runs/protocol-c-current/fold0-v2-dd79473 \
      --out results/tables/v2-sweep
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from pathlib import Path


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def row_metrics(summary):
    if summary is None:
        return None
    sub = summary.get("subgroups") or {}
    out = dict(acc=summary["accuracy"], auc=summary["auc"], bce=summary.get("image_loss"),
               fp=len(summary.get("genuine_false_positives", [])), misses=len(summary.get("altered_misses", [])),
               n_genuine=summary.get("n_genuine"), n_altered=summary.get("n_altered"), epoch=summary.get("selected_epoch"),
               n_unavailable=summary.get("n_context_unavailable"))
    for name, value in sub.items():
        if isinstance(value, dict):
            out[f"{name}_auc"] = value.get("auc")
            out[f"{name}_acc"] = value.get("accuracy")
            out[f"{name}_n"] = value.get("n_altered")
    return out


def collect(pattern, unit):
    rows = []
    for root in sorted(glob.glob(pattern), key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]):
        root = Path(root)
        match = re.search(rf"{unit}(\d+)", root.name)
        index = int(match.group(1)) if match else root.name
        entry = dict(unit=index, root=str(root))
        for arm in ("off", "on"):
            entry[f"val_{arm}"] = row_metrics(load(root / "training" / arm / "val_summary.json"))
            entry[f"test_{arm}"] = row_metrics(load(root / "evaluation" / arm / "test_summary.json"))
        rows.append(entry)
    return rows


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.)


def table(rows, phase, unit, families, markdown=True):
    header = [unit, "arm", "epoch", "acc", "AUC", "BCE", "FP", "misses"] + [f"{f} AUC" for f in families]
    lines = []
    for row in rows:
        for arm in ("off", "on"):
            m = row.get(f"{phase}_{arm}")
            if m is None:
                continue
            cells = [str(row["unit"]), arm.upper(), fmt(m.get("epoch")), fmt(m["acc"]), fmt(m["auc"]), fmt(m.get("bce")),
                     fmt(m["fp"]), fmt(m["misses"])] + [fmt(m.get(f"{f}_auc")) for f in families]
            lines.append(cells)
    for arm in ("off", "on"):
        present = [row[f"{phase}_{arm}"] for row in rows if row.get(f"{phase}_{arm}") is not None]
        if len(present) > 1:
            cells = ["mean±sd", arm.upper(), "-"]
            for key in ("acc", "auc", "bce"):
                mu, sd = mean_sd([p.get(key) for p in present])
                cells.append(f"{fmt(mu)}±{fmt(sd)}" if mu is not None else "-")
            cells += [fmt(sum(p["fp"] for p in present)), fmt(sum(p["misses"] for p in present))]
            for f in families:
                mu, sd = mean_sd([p.get(f"{f}_auc") for p in present])
                cells.append(f"{fmt(mu)}±{fmt(sd)}" if mu is not None else "-")
            lines.append(cells)
    if markdown:
        text = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
        text += "".join("| " + " | ".join(cells) + " |\n" for cells in lines)
    else:
        text = "\\begin{tabular}{" + "l" * 3 + "r" * (len(header) - 3) + "}\n\\toprule\n"
        text += " & ".join(header).replace("±", "$\\pm$") + " \\\\\n\\midrule\n"
        text += "".join(" & ".join(cells).replace("±", "$\\pm$") + " \\\\\n" for cells in lines)
        text += "\\bottomrule\n\\end{tabular}\n"
    return text


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=None, help="glob of Protocol A study roots")
    parser.add_argument("--b", default=None, help="glob of Protocol B study roots")
    parser.add_argument("--c", default=None, help="Protocol C scoring root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = {}
    md = []
    tex = []
    if args.a:
        rows = collect(args.a, "fold")
        digest["A"] = rows
        for phase, title in (("val", "Protocol A, selected source validation"), ("test", "Protocol A, official test")):
            md.append(f"## {title}\n\n" + table(rows, phase, "fold", ["copy", "inpainting"]))
            tex.append(f"% {title}\n" + table(rows, phase, "fold", ["copy", "inpainting"], markdown=False))
    if args.b:
        rows = collect(args.b, "group")
        digest["B"] = rows
        for phase, title in (("val", "Protocol B, selected source validation"), ("test", "Protocol B, held-out nationality queries")):
            md.append(f"## {title}\n\n" + table(rows, phase, "group", ["copy", "inpainting"]))
            tex.append(f"% {title}\n" + table(rows, phase, "group", ["copy", "inpainting"], markdown=False))
    if args.c:
        rows = collect(args.c, "fold")
        digest["C"] = rows
        md.append("## Protocol C, FantasyID locked queries\n\n" + table(rows, "test", "source", ["face_swap", "text_inpaint"]))
        tex.append("% Protocol C\n" + table(rows, "test", "source", ["face_swap", "text_inpaint"], markdown=False))
    (out / "tables.md").write_text("\n".join(md))
    (out / "tables.tex").write_text("\n".join(tex))
    (out / "digest.json").write_text(json.dumps(digest, indent=2, sort_keys=True))
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
