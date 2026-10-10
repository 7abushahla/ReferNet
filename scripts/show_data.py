#!/usr/bin/env python3
"""Visual checks of data/processed: references, zone masks, and query/reference/mask examples.
Writes results/figures/data_check_{references,zones,pairs}.jpg."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.data.references import reference_path  # noqa: E402

P = Path("data/processed")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
rec = pd.read_csv(P / "records.csv").fillna("")


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def grid(tiles, ncol, scale=0.4):
    tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles]
    h, w = tiles[0].shape[:2]
    while len(tiles) % ncol:
        tiles.append(np.full((h, w, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)]
    return np.vstack(rows)


designs = sorted(rec.design_id.unique())
# 1. references: median over all genuine, per design
tiles = [label(cv2.imread(str(P / "references" / f"{d.replace('/', '_')}_all.jpg")), d) for d in designs]
cv2.imwrite(str(OUT / "data_check_references.jpg"), grid(tiles, 4), [cv2.IMWRITE_JPEG_QUALITY, 85])

# 2. zone masks over the references
tiles = []
for d in designs:
    ref = cv2.imread(str(P / "references" / f"{d.replace('/', '_')}_all.jpg"))
    z = cv2.imread(str(P / "zones" / f"{d.replace('/', '_')}.png"), 0)
    ov = ref.copy()
    ov[z > 0] = (0.45 * ov[z > 0] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
    tiles.append(label(ov, f"{d} zones"))
cv2.imwrite(str(OUT / "data_check_zones.jpg"), grid(tiles, 4), [cv2.IMWRITE_JPEG_QUALITY, 85])

# 3. query, reference (opposite fold), |query - reference|, ground-truth mask: one SIDTD, one FantasyID text, one FantasyID face
picks = [rec[(rec.dataset == "sidtd") & (rec.attack_family == "text_copy")].iloc[3],
         rec[(rec.dataset == "sidtd") & (rec.attack_family == "text_inpaint")].iloc[7],
         rec[(rec.dataset == "fantasyid") & (rec.attack_source.str.startswith("attack3")) & (rec.capture == "scan")].iloc[2],
         rec[(rec.dataset == "fantasyid") & (rec.attack_family == "face_swap") & (rec.capture == "huawei")].iloc[1],
         rec[(rec.dataset == "fantasyid") & (rec.label == 0) & (rec.path.str.startswith("fantasyid/test")) & (rec.capture == "scan")].iloc[5]]
rows = []
for r in picks:
    q = cv2.imread(str(P / "canvases" / f"{r.image_id}.jpg"))
    ref = cv2.imread(str(reference_path(P / "references", r.design_id, r.identity_id)))
    diff = cv2.absdiff(q, ref)
    diff = cv2.cvtColor(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    diff = np.clip(diff.astype(np.float32) * 3, 0, 255).astype(np.uint8)
    mpath = P / "masks" / f"{r.image_id}.png"
    m = cv2.imread(str(mpath), 0) if mpath.is_file() else np.zeros(q.shape[:2], np.uint8)
    mo = q.copy()
    mo[m > 0] = (0.4 * mo[m > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    tag = f"{r.dataset} {r.attack_family or 'genuine'} [{r.altered_fields or '-'}]"
    rows.append(grid([label(q, "query: " + tag), label(ref, "reference: design median, other fold"),
                      label(diff, "|query - reference| x3"), label(mo, "ground-truth box mask" if r.label else "genuine: no mask")], 4, 0.5))
cv2.imwrite(str(OUT / "data_check_pairs.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", sorted(p.name for p in OUT.glob("data_check_*.jpg")))
