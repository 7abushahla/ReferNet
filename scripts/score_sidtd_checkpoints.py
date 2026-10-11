#!/usr/bin/env python3
"""Diagnostic (spec Section 6, item 2): score the SIDTD paper's released template checkpoints on the paper's
10 fold test files with the paper's own preprocessing, so the published Table 5 rows can be reproduced on our
copy of the data. Follows SIDTD/models/Baseline/{test_kfold_baseline,_utils,dataset/custom_dataset}.py:
cv2.imread, BGR to RGB, resize to 299 x 299 (bilinear), ImageNet mean/std, accuracy by argmax, ROC AUC of the
softmax probability of "fake" (label 1). Also scores a compression-equalized copy (every test file re-encoded
at JPEG quality 75 before preprocessing).

    python scripts/score_sidtd_checkpoints.py --models resnet50[,efficientnet-b3] --raw data/raw/sidtd \
        --ckpt-dir data/raw/sidtd/paper_checkpoints --out results/tables/sidtd_checkpoints.md

Checkpoints: http://datasets.cvc.uab.es/SIDTD/balanced_templates_SIDTD/<model>_trained_models.zip, one .pth per
fold named *_n<k>.pth. Download and unzip them under --ckpt-dir/<model>/ first (scripts/cluster/get_sidtd_ckpts.sh).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = {"resnet50": 299, "efficientnet-b3": 299}
PUBLISHED = {"resnet50": (98.1, 1.2, 0.99, 0.01), "efficientnet-b3": (99.4, 1.0, 1.00, 0.00)}


def fold_rows(raw: Path, k: int) -> list[tuple[Path, int]]:
    """(image path, label) for the paper's test file of fold k; label 1 = fake."""
    import pandas as pd
    p = pd.read_csv(raw / "kfold_split_templates" / "split_kfold" / f"test_split_SIDTD_it_{k}.csv")
    return [(raw / path, int(label)) for path, label in zip(p.image_path, p.label)]


def preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)   # albumentations Resize default
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def pick_checkpoint(files, k: int) -> str:
    for f in files:
        if re.search(rf"_n{k}\.pth$", str(f)):
            return str(f)
    raise FileNotFoundError(f"no checkpoint for fold {k} among {len(list(files))} files")


def build(model_name: str):
    import torch.nn as nn
    if model_name == "resnet50":
        from torchvision import models
        m = models.resnet50(weights=None)
        m.avgpool = nn.AdaptiveAvgPool2d(1)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    if model_name == "efficientnet-b3":
        from efficientnet_pytorch import EfficientNet
        m = EfficientNet.from_name("efficientnet-b3")
        m._fc = nn.Linear(m._fc.in_features, 2)
        return m
    raise ValueError(model_name)


def _jpeg(bgr: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def score_fold(model, rows, size, device, equalize_qf=None, batch=32):
    import torch
    from sklearn.metrics import roc_auc_score
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            xs = []
            for p, _ in chunk:
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"cannot read {p}")
                if equalize_qf is not None:
                    bgr = _jpeg(bgr, equalize_qf)
                xs.append(preprocess(bgr, size))
            x = torch.from_numpy(np.stack(xs)).to(device)
            logits = model(x).float()
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            labels.extend(l for _, l in chunk)
    labels = np.asarray(labels)
    return dict(accuracy=float((np.asarray(preds) == labels).mean()), auc=float(roc_auc_score(labels, probs)),
                n=int(len(labels)))


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="resnet50")
    ap.add_argument("--raw", default="data/raw/sidtd")
    ap.add_argument("--ckpt-dir", default="data/raw/sidtd/paper_checkpoints")
    ap.add_argument("--folds", default="0-9")
    ap.add_argument("--out", default="results/tables/sidtd_checkpoints.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    raw, ck = Path(a.raw), Path(a.ckpt_dir)
    lo, hi = (int(x) for x in a.folds.split("-"))
    device = torch.device(a.device)
    lines = ["# SIDTD released template checkpoints scored on our copy of the paper's folds\n",
             "| Model | Published acc (%) | Published AUC | Ours, raw: acc (%) | Ours, raw: AUC | Equalized QF75: acc (%) | Equalized QF75: AUC | folds |",
             "|---|---|---|---|---|---|---|---|"]
    per_fold = {}
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        files = sorted((ck / name).rglob("*.pth"))
        model = build(name).to(device)
        res = {"raw": [], "eq": []}
        for k in range(lo, hi + 1):
            sd = torch.load(pick_checkpoint(files, k), map_location="cpu")
            model.load_state_dict(sd if "state_dict" not in sd else sd["state_dict"])
            rows = fold_rows(raw, k)
            r_raw = score_fold(model, rows, SIZE[name], device)
            r_eq = score_fold(model, rows, SIZE[name], device, equalize_qf=75)
            res["raw"].append(r_raw)
            res["eq"].append(r_eq)
            print(f"[{name} fold {k}] raw acc {r_raw['accuracy']:.4f} auc {r_raw['auc']:.4f} | "
                  f"eq75 acc {r_eq['accuracy']:.4f} auc {r_eq['auc']:.4f} | n {r_raw['n']}", flush=True)
        per_fold[name] = res

        def ms(key, which, scale=1.0):
            v = np.array([r[key] for r in res[which]]) * scale
            return f"{v.mean():.{1 if scale == 100 else 3}f} ± {v.std():.{1 if scale == 100 else 3}f}"
        pa, psa, pu, psu = PUBLISHED[name]
        lines.append(f"| {name} | {pa:.1f} ± {psa:.1f} | {pu:.2f} ± {psu:.2f} | {ms('accuracy', 'raw', 100)} | "
                     f"{ms('auc', 'raw')} | {ms('accuracy', 'eq', 100)} | {ms('auc', 'eq')} | {len(res['raw'])} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(per_fold, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
