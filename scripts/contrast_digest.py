#!/usr/bin/env python3
"""Digest the fold-0 recipe contrast (OFF and the ON variants) from training receipts, metadata only.

Each NAME=ROOT argument points at a study root whose training/<arm>/val_summary.json and
val_predictions.csv exist (written when training finishes). Reports the selected epoch, accuracy,
AUC, copy and inpainting AUC, genuine false positives, altered misses, and the AUC of the query
logit alone (the global path scored without the correction), which is the branch-collapse check.

Usage:
  python scripts/contrast_digest.py --out results/tables/v2-sweep/contrast.json \
      off=results/runs/protocol-a-current/fold0-v2-dd79473:off on_sharedbn_aux=...fold0-v2-dd79473:on \
      on_storedbn=...fold0-v2sbnnoaux-dd79473:on on_storedbn_aux=...fold0-v2sbn-dd79473:on
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rank_auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks, index = {}, 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        for position in range(index, stop + 1):
            ranks[position] = (index + stop) / 2 + 1
        index = stop + 1
    positive_rank_sum = sum(ranks[position] for position, (unused, label) in enumerate(pairs) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def digest(root, arm):
    directory = Path(root) / "training" / arm
    summary = json.loads((directory / "val_summary.json").read_text())
    with (directory / "val_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [int(row["label"]) for row in rows]
    query = [float(row["query_logit"]) for row in rows]
    sub = summary.get("subgroups") or {}
    return dict(root=str(root), arm=arm, epoch=summary.get("selected_epoch"), accuracy=summary["accuracy"], auc=summary["auc"],
                copy_auc=(sub.get("copy") or {}).get("auc"), inpainting_auc=(sub.get("inpainting") or {}).get("auc"),
                fp=len(summary.get("genuine_false_positives", [])), misses=len(summary.get("altered_misses", [])),
                n_genuine=summary.get("n_genuine"), n_altered=summary.get("n_altered"), query_only_auc=rank_auc(query, labels),
                n=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("runs", nargs="+", help="NAME=ROOT:ARM")
    args = parser.parse_args(argv)
    out = {}
    for item in args.runs:
        name, spec = item.split("=", 1)
        root, arm = spec.rsplit(":", 1)
        out[name] = digest(root, arm)
        print(name, json.dumps(out[name]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
