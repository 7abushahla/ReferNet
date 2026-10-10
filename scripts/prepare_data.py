#!/usr/bin/env python3
"""Build data/processed from data/raw for the datasets present (SIDTD, FantasyID; IDNet when staged).

Outputs under data/processed:
    records.csv, transforms.json (letterbox then registration, composed), registration.csv,
    canvases/<image_id>.png, valid/<image_id>.png (only when padded or warped out of frame),
    masks/<image_id>.png (altered only), zones/<design>.png, references/<design>_{fold0,fold1,all}.png,
    splits/protocol_a_it{0..9}.csv, splits/protocol_b_g{0..9}.csv, splits/protocol_c.csv, counts.md
Usage: python scripts/prepare_data.py [--data-root data] [--datasets sidtd,fantasyid] [--workers 4] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.data.forgeries import build_masks  # noqa: E402
from refer_or_clear.data.records import build_records, provenance_components  # noqa: E402
from refer_or_clear.data.rectify import CANVAS_EXT, to_canvas  # noqa: E402
from refer_or_clear.data.register import register_design, warp  # noqa: E402
from refer_or_clear.data.references import build_references, build_specimens, build_zone_masks  # noqa: E402
from refer_or_clear.data.splits import protocol_a_kfold, protocol_b_groups, protocol_c  # noqa: E402


def md(df) -> str:
    """Minimal markdown table without optional dependencies."""
    import pandas as pd
    if isinstance(df, pd.Series):
        df = df.to_frame("n")
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def _canvas_job(args):
    image_id, src, dst_img, dst_valid, R = args
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {src}")
    canvas, valid, T = to_canvas(img)
    if R is not None:
        R = np.asarray(R)
        canvas, valid, T = warp(canvas, R), warp(valid, R), R @ T
    cv2.imwrite(str(dst_img), canvas)
    if dst_valid.is_file():
        dst_valid.unlink()
    if valid.min() == 0:
        cv2.imwrite(str(dst_valid), valid)
    return image_id, T.tolist()


def _register_job(args):
    design_id, image_ids, genuine_ids, canvas_dir = args
    return register_design(design_id, image_ids, genuine_ids, canvas_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", default="sidtd,fantasyid")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    raw, out = Path(a.data_root) / "raw", Path(a.data_root) / "processed"
    for d in ("canvases", "valid", "masks", "zones", "references", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)
    stale_jpg = list((out / "canvases").glob("*.jpg")) + list((out / "references").glob("*.jpg"))
    if stale_jpg:
        print(f"      removing {len(stale_jpg)} JPEG canvases and references from the old pipeline")
        for p in stale_jpg:
            p.unlink()
        a.force = True

    print("[1/7] records")
    rec = build_records(raw, tuple(a.datasets.split(",")))
    rec["component_id"] = provenance_components(rec)
    rec.to_csv(out / "records.csv", index=False)
    print(f"      {len(rec)} images, {rec.identity_id.nunique()} identities, {rec.component_id.nunique()} components")

    print("[2/7] letterbox canvases")
    tpath, rpath = out / "transforms.json", out / "registration.csv"
    stale = a.force or not rpath.is_file()
    transforms = {} if stale else json.load(open(tpath))
    jobs = [(r.image_id, raw / r.path, out / "canvases" / f"{r.image_id}{CANVAS_EXT}", out / "valid" / f"{r.image_id}.png", None)
            for r in rec.itertuples(index=False)
            if stale or r.image_id not in transforms or not (out / "canvases" / f"{r.image_id}{CANVAS_EXT}").is_file()]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (image_id, T) in enumerate(ex.map(_canvas_job, jobs, chunksize=16), 1):
            transforms[image_id] = T
            if i % 1000 == 0:
                print(f"      {i}/{len(jobs)}")

    print("[3/7] registration to the design (SIFT + RANSAC similarity, two passes)")
    if jobs or stale:
        rjobs = [(d, g.image_id.tolist(), g[g.label == 0].image_id.tolist(), out / "canvases")
                 for d, g in rec.groupby("design_id") if (g.label == 0).any()]
        with ProcessPoolExecutor(a.workers) as ex:
            reg = pd.concat(list(ex.map(_register_job, rjobs)), ignore_index=True)
        Rmap = {r.image_id: r.R for r in reg.itertuples(index=False) if r.applied}
        jobs = [(r.image_id, raw / r.path, out / "canvases" / f"{r.image_id}{CANVAS_EXT}", out / "valid" / f"{r.image_id}.png",
                 Rmap.get(r.image_id)) for r in rec.itertuples(index=False)]
        with ProcessPoolExecutor(a.workers) as ex:
            for image_id, T in ex.map(_canvas_job, jobs, chunksize=16):
                transforms[image_id] = T
        reg.drop(columns=["R"]).to_csv(rpath, index=False)
    json.dump(transforms, open(tpath, "w"))
    reg = pd.read_csv(rpath)
    print(f"      applied to {int(reg.applied.sum())}/{len(reg)}; inliers min {reg.inliers.min()}, "
          f"|rot| max {reg.rot_deg.abs().max():.2f} deg, shift max {reg.shift_px.max():.1f} px")
    T = {k: np.array(v) for k, v in transforms.items()}

    print("[4/7] zone masks")
    zones = build_zone_masks(rec, raw, T, out / "zones")
    print("[5/7] references")
    refs = build_references(rec, out / "canvases", out / "references")
    specimens = build_specimens(rec, raw / "idnet_templates", out / "references")
    print(f"      specimens: {len(specimens)} designs")
    print("[6/7] ground-truth masks")
    masks = build_masks(rec, raw, T, out / "masks")
    print("[7/7] splits and per-fold references")
    for old in (out / "splits").glob("protocol_a*.csv"):
        old.unlink()
    folds_a = protocol_a_kfold(rec, raw)
    fold_refs = []
    for k, df in folds_a.items():
        df.to_csv(out / "splits" / f"protocol_a_it{k}.csv", index=False)
        train_ids = set(df[df.role == "train"].image_id)
        fold_refs.append(build_references(rec[rec.image_id.isin(train_ids)], out / "canvases", out / "references",
                                          group=f"it{k}", write_all=False))
    fold_refs = pd.concat(fold_refs, ignore_index=True)
    pa = pd.concat(folds_a.values(), ignore_index=True)
    for old in (out / "splits").glob("protocol_b_g*.csv"):
        old.unlink()
    groups_b = protocol_b_groups(rec)
    for k, df in groups_b.items():
        df.to_csv(out / "splits" / f"protocol_b_g{k}.csv", index=False)
    pbg = pd.concat(groups_b.values(), ignore_index=True)
    pc = protocol_c(rec)
    pc.to_csv(out / "splits" / "protocol_c.csv", index=False)
    (out / "splits" / "protocol_b.csv").unlink(missing_ok=True)

    L = ["# Data counts (generated by scripts/prepare_data.py)\n", "## Images per dataset, design and label\n",
         md(rec.groupby(["dataset", "design_id", "label"]).size().unstack(fill_value=0)),
         "\n\n## Attack families\n", md(rec[rec.label == 1].groupby(["dataset", "attack_family"]).size()),
         "\n\n## Registration (per design: images registered, min inliers, rotation and shift spread)\n",
         md(reg.groupby("design_id").agg(n=("image_id", "size"), applied=("applied", "sum"), inliers_min=("inliers", "min"),
                                         rot_sd_deg=("rot_deg", "std"), rot_max_deg=("rot_deg", lambda s: s.abs().max()),
                                         shift_med_px=("shift_px", "median"), shift_max_px=("shift_px", "max"))),
         "\n\n## References\n", md(refs.set_index("design_id")),
         "\n\n## Protocol A per-fold references (training files only)\n",
         md(fold_refs.groupby("group").agg(designs=("design_id", "size"), min_half=("n_fold0", "min"),
                                           min_half1=("n_fold1", "min"), min_coverage=("min_coverage", "min"))),
         "\n\n## Blank specimens (IDNet paper templates registered to the design median)\n", md(specimens.set_index("design_id")), "\n\n## Zone masks\n", md(zones.set_index("design_id")),
         "\n\n## Ground-truth masks (altered images)\n",
         md(masks.groupby("dataset").agg(n=("image_id", "size"), mean_fraction=("mask_fraction", "mean"),
                                         zero_rects=("n_altered_rects", lambda s: int((s == 0).sum()))))]
    for name, df in (("Protocol A (all folds stacked)", pa),
                     ("Protocol B, held-out nationalities (all groups stacked)", pbg), ("Protocol C (FantasyID)", pc)):
        keys = ["ref_group", "role", "label"] if "ref_group" in df and df.ref_group.astype(bool).any() else ["role", "subset", "label"]
        L += [f"\n\n## {name}\n", md(df.groupby(keys).size().unstack(fill_value=0))]
    (out / "counts.md").write_text("\n".join(L) + "\n")
    print(open(out / "counts.md").read())


if __name__ == "__main__":
    main()
