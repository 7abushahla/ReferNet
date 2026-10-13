"""Training-free alignment-residual diagnostic for copy-and-replace against a design reference.

For every query whose reference context is available, each connected component of the
personal-data zone (a field box) is phase-correlated against the registered design median.
A pasted field crop from another document of the same design (SIDTD crop-and-replace,
shifted by up to 10 px) carries a background texture that disagrees locally with the global
registration; genuine and inpainted fields do not. The diagnostic has no trained parameters
and no design knowledge, so it is design-agnostic by construction.

Scores per query (higher = more suspicious):
  shift_max   max over field boxes of the local phase-correlation shift magnitude (px)
  resid_max   max over boxes of 1 - NCC(query, reference) at zero shift, within the box
  resid_gain  max over boxes of NCC improvement after local re-alignment (NCC_best - NCC_0)
Reported: AUC per score overall and per attack family, with the reference-conditioned ON
arm's own scores for fusion when test_predictions.csv is present.

Diagnostic only: run from a scratch copy on a compute node; never a selection input.

Usage:
  python alignment_residual_diagnostic.py --protocol b --request REQUEST.json --role test \
      --registration <study>/evaluation/registration.json --authorization <study>/evaluation/authorization.json \
      --out OUT_DIR [--predictions <study>/evaluation/on/test_predictions.csv]
  python alignment_residual_diagnostic.py --protocol a --request REQUEST.json --role val --out OUT_DIR
      (val: builds its own registration table under OUT_DIR)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
import yaml

from refer_or_clear.data.dataset import IMAGENET_MEAN, IMAGENET_STD

MARGIN = 12
MIN_BOX = 12


def denormalize_gray(tensor):
    rgb = tensor[:3].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    gray = cv2.cvtColor(np.clip(rgb * 255., 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.


def shifted(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def boxes(zone):
    count, unused, stats, unused_centroids = cv2.connectedComponentsWithStats((zone > 0).astype(np.uint8), connectivity=8)
    out = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if w >= MIN_BOX and h >= MIN_BOX:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def score_pair(query_gray, reference_gray, zone, overlap):
    rows = []
    h, w = query_gray.shape
    for x, y, bw, bh in boxes(zone):
        x0, y0 = max(0, x - MARGIN), max(0, y - MARGIN)
        x1, y1 = min(w, x + bw + MARGIN), min(h, y + bh + MARGIN)
        if (overlap[y0:y1, x0:x1] == 0).mean() > .05:
            continue
        q = query_gray[y0:y1, x0:x1]
        r = reference_gray[y0:y1, x0:x1]
        window = cv2.createHanningWindow((q.shape[1], q.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(q, r, window)
        ncc0 = ncc(q, r)
        ncc1 = ncc(shifted(q, dx, dy), r)
        rows.append(dict(x=x, y=y, w=bw, h=bh, dx=float(dx), dy=float(dy), shift=float(np.hypot(dx, dy)),
                         response=float(response), ncc0=ncc0, ncc1=ncc1))
    if not rows:
        return dict(n_boxes=0, shift_max=0., resid_max=0., resid_gain=0.), rows
    return dict(n_boxes=len(rows), shift_max=max(r["shift"] for r in rows), resid_max=max(1. - r["ncc0"] for r in rows),
                resid_gain=max(r["ncc1"] - r["ncc0"] for r in rows)), rows


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    combined = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and combined[j + 1] == combined[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2. + 1.
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.) / (len(pos) * len(neg)))


def load_context(args, request):
    if args.protocol == "b":
        from refer_or_clear.data import protocol_b_reference as bridge
        evaluation = dict(path=str(args.authorization), sha256=sha256(args.authorization)) if args.authorization else None
        context = bridge.load_reference_context(Path(request["records"]["path"]), Path(request["split"]["path"]),
            group=request["group"], records_sha256=request["records"]["sha256"], target_binding=request["target_enrollment"],
            enrollment_binding=request["enrollment"], raw_root=Path(request["raw_root"]), evaluation_binding=evaluation)
        return bridge, context
    from refer_or_clear.data import protocol_a_reference as bridge
    relocation = {} if "metadata_relocation" not in request else {"metadata_relocation_binding": request["metadata_relocation"]}
    evaluation = dict(path=str(args.authorization), sha256=sha256(args.authorization)) if args.authorization else None
    context = bridge.load_reference_context(Path(request["records"]["path"]), Path(request["split"]["path"]),
        fold=request["fold"], records_sha256=request["records"]["sha256"], enrollment_binding=request["enrollment"],
        root_binding=request["root"], raw_root=Path(request["raw_root"]), evaluation_binding=evaluation, **relocation)
    return bridge, context


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["a", "b"], required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--role", choices=["val", "test"], required=True)
    parser.add_argument("--registration", default=None, help="existing registration table (required for test)")
    parser.add_argument("--authorization", default=None, help="evaluation/authorization.json (required for test)")
    parser.add_argument("--predictions", default=None, help="ON arm test_predictions.csv for fusion")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(args.request).read_text())
    config = yaml.safe_load(Path(request["config"]["path"]).read_bytes())
    bridge, context = load_context(args, request)
    if args.role == "test":
        assert args.registration and args.authorization, "test role needs the study's registration and authorization"
        binding = dict(path=str(args.registration), sha256=sha256(args.registration))
        loader = getattr(bridge, f"load_protocol_{args.protocol}_registration")
        table = loader(context, binding, roles=["test"])
    else:
        builder = getattr(bridge, f"build_protocol_{args.protocol}_registration")
        table = builder(context, roles=["val"], output_path=out / "registration.json")
    dataset = getattr(bridge, f"Protocol{args.protocol.upper()}ReferenceDataset")(
        context, table, role=args.role, reference_enabled=True, augment=dict(config["augment"], enabled=False), cache_bytes=0)
    rows = []
    details = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = sample["meta"]
        record = dict(image_id=meta["image_id"], design_id=meta["design_id"], attack_family=meta["attack_family"],
                      label=int(sample["label"]), available=bool(sample["context_available"]))
        if record["available"]:
            query_gray = denormalize_gray(sample["query"])
            reference_gray = denormalize_gray(sample["reference"])
            zone = sample["query"][3].numpy()
            overlap = sample["overlap"][0].numpy()
            scores, boxes_rows = score_pair(query_gray, reference_gray, zone, overlap)
            record.update(scores)
            details[meta["image_id"]] = boxes_rows
        else:
            record.update(n_boxes=0, shift_max=0., resid_max=0., resid_gain=0.)
        rows.append(record)
        if index % 50 == 0:
            print(f"{index}/{len(dataset)}", flush=True)
    fields = ["image_id", "design_id", "attack_family", "label", "available", "n_boxes", "shift_max", "resid_max", "resid_gain"]
    with (out / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "boxes.json").write_text(json.dumps(details, sort_keys=True))
    on_scores = None
    if args.predictions:
        with open(args.predictions, newline="") as handle:
            on_scores = {row["image_id"]: float(row.get("altered_score") or row.get("score") or 1. - float(row["bonafide_score"]))
                         for row in csv.DictReader(handle)}
    avail = [r for r in rows if r["available"]]
    families = sorted({r["attack_family"] for r in avail if r["label"] == 1})
    summary = dict(n=len(rows), n_available=len(avail), families=families, auc={})
    genuine = [r for r in avail if r["label"] == 0]
    for key in ("shift_max", "resid_max", "resid_gain"):
        entry = dict(all=auc([r[key] for r in avail], [r["label"] for r in avail]))
        for family in families:
            subset = genuine + [r for r in avail if r["attack_family"] == family]
            entry[family] = auc([r[key] for r in subset], [r["label"] for r in subset])
        summary["auc"][key] = entry
    if on_scores:
        def z(values):
            values = np.asarray(values, float)
            return (values - values.mean()) / (values.std() + 1e-9)
        on = [on_scores[r["image_id"]] for r in avail]
        entry = dict(all=auc(on, [r["label"] for r in avail]))
        fused = z(on) + z([r["shift_max"] for r in avail])
        fusion = dict(all=auc(fused, [r["label"] for r in avail]))
        for family in families:
            mask = [r["label"] == 0 or r["attack_family"] == family for r in avail]
            labels = [r["label"] for r, m in zip(avail, mask) if m]
            entry[family] = auc([s for s, m in zip(on, mask) if m], labels)
            fusion[family] = auc([s for s, m in zip(fused, mask) if m], labels)
        summary["auc"]["on_arm"] = entry
        summary["auc"]["on_plus_shift_z"] = fusion
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
