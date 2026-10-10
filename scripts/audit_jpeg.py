#!/usr/bin/env python3
"""JPEG quality audit: estimate the encoder quality of every source image from its luminance
quantization table (IJG scaling against the standard table), cross-tabulate by dataset, folder,
split and label, and measure how well compression alone separates the classes (DeepID reported
QF 95 bona fide vs QF 75 altered in FantasyID's train-val release). Also measures an 8-pixel
blockiness statistic on the raw files and on the re-encoded canvases, to show whether the
letterbox rescale plus QF 95 re-encode removes the pixel-level trace. Writes
results/tables/jpeg_audit.md."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

STD_LUMA = np.array([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
                     18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99], float)


def estimate_quality(path: Path) -> float | None:
    try:
        q = Image.open(path).quantization
    except Exception:
        return None
    if not q:
        return None
    scale = np.median(100.0 * np.array(q[0], float) / STD_LUMA)   # IJG: scale = 5000/Q for Q<50, else 200-2Q
    return round(float(5000.0 / scale if scale > 100 else (200.0 - scale) / 2.0), 1)


def blockiness(path: Path) -> float:
    """Mean gradient across 8-pixel column boundaries over mean gradient elsewhere (1.0 = no grid trace)."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    d = np.abs(np.diff(g, axis=1))
    return float(d[:, 7::8].mean() / (np.delete(d, np.s_[7::8], axis=1).mean() + 1e-6))


def sym_auc(y, x) -> float:
    a = roc_auc_score(y, x)
    return round(max(a, 1 - a), 3)


def main():
    raw, proc = Path("data/raw"), Path("data/processed")
    rec = pd.read_csv(proc / "records.csv").fillna("")
    rec["qf"] = [estimate_quality(raw / p) for p in rec.path]
    rec["folder"] = rec.path.str.split("/").str[1:3].str.join("/")
    g = rec.groupby(["dataset", "folder", "label"]).qf.agg(n="size", median="median", min="min", max="max").reset_index()
    out = ["# JPEG quality audit (estimated from quantization tables)\n",
           "| dataset | folder | label | n | median QF | min | max |", "|---|---|---|---|---|---|---|"]
    out += [f"| {r.dataset} | {r.folder} | {r.label} | {r.n} | {r['median']} | {r['min']} | {r['max']} |" for _, r in g.iterrows()]
    out += ["\n## Shortcut check per split (AUC of 'lower QF means altered'; 1.0 = compression alone separates the classes)\n",
            "| protocol | role | subset | n | median QF genuine | median QF altered | AUC(QF) | acc QF<85 |", "|---|---|---|---|---|---|---|---|"]
    for name in ("protocol_a", "protocol_b"):
        s = pd.read_csv(proc / "splits" / f"{name}.csv").fillna("").merge(rec[["image_id", "qf"]], on="image_id")
        for (role, subset), gg in s.groupby(["role", "subset"]):
            if gg.label.nunique() < 2:
                continue
            out.append(f"| {name} | {role} | {subset or '(all)'} | {len(gg)} | {gg[gg.label==0].qf.median()} | {gg[gg.label==1].qf.median()} | "
                       f"{roc_auc_score(gg.label, -gg.qf):.3f} | {((gg.qf < 85).astype(int) == gg.label).mean():.3f} |")
    # pixel-level trace on the Protocol B held-out test subset: raw files vs re-encoded canvases
    s = pd.read_csv(proc / "splits" / "protocol_b.csv").fillna("")
    s = s[(s.role == "test") & (s.subset == "heldout")].merge(rec[["image_id", "path"]], on="image_id")
    b_raw = [blockiness(raw / p) for p in s.path]
    b_can = [blockiness(proc / "canvases" / f"{i}.jpg") for i in s.image_id]
    out += ["\n## Pixel-level trace, Protocol B held-out test (n = %d)\n" % len(s),
            "| input | AUC of 8-px blockiness |", "|---|---|",
            f"| raw files (native resolution, native QF) | {sym_auc(s.label, b_raw)} |",
            f"| canvases (letterbox 512x800, re-encoded QF 95) | {sym_auc(s.label, b_can)} |"]
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/jpeg_audit.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
