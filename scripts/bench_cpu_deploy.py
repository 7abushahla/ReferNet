#!/usr/bin/env python3
"""Inference cost of the deployment model (SpecimenNet OFF and ON, the fold-0 pair), batch one, FP32.

The deployment path is timed stage by stage on raw SIDTD test files, with the design library (reference median, its
SIFT features, validity and zone masks) loaded once per design outside the timed region, as a counter system would
hold it:

  decode        read the raw file (cv2.imread)
  canvas        letterbox onto the 512x800 canvas (rectify.to_canvas)
  registration  SIFT features of the query canvas, RANSAC similarity to the design reference, overlap and zone warp
                (the same functions as the study's registration table builder)
  reference     warp the design median into the query frame (ON only)
  tensor        ImageNet normalization and tensor assembly
  forward OFF   single-image network (query path alone)
  forward ON    the same network with the reference pass and the local correction
  end to end    decode + canvas + registration + tensor + forward (ON also includes reference)

Median and p95 over --n documents after --warmup warm-ups; documents per second from the median end-to-end time;
parameter count and checkpoint size. The tensor transfer to the device is outside the timed forward. Writes results/timing/deploy-<host>-<device>.json.

    python scripts/bench_cpu_deploy.py --off <off/best.pt> --on <on/best.pt> --weights <weights.pth> [--n 200 --threads 4]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.data.dataset import _normalize  # noqa: E402
from refer_or_clear.data.enrollment import _fit, _similarity, _warp  # noqa: E402
from refer_or_clear.data.direct_reference import _overlap, _available  # noqa: E402
from refer_or_clear.data.rectify import to_canvas  # noqa: E402
from refer_or_clear.data.register import _features  # noqa: E402
from refer_or_clear.models.direct_reference import build_direct_reference_model  # noqa: E402

WEIGHTS_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def load_arm(path, weights, arm):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    assert cfg["arm"] == arm, (cfg["arm"], arm)
    recipe = dict(local_branch=True, reference_bn="shared", fused_query_weight=1.)
    recipe.update({k: cfg[k] for k in recipe if k in cfg})
    if isinstance(cfg.get("arm_recipe"), dict) and isinstance(cfg["arm_recipe"].get(arm), dict):
        recipe.update({k: v for k, v in cfg["arm_recipe"][arm].items() if k in recipe})
    local = bool(recipe["local_branch"]) if arm == "on" else False
    model = build_direct_reference_model(Path(weights), WEIGHTS_SHA256, reference_enabled=arm == "on", head_seed=42,
                                         adapter_seed=42, local_enabled=local, reference_bn=recipe["reference_bn"],
                                         fused_query_weight=float(recipe["fused_query_weight"]))
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dict(epoch=ck["epoch"], local_enabled=local, reference_bn=recipe["reference_bn"],
                       fused_query_weight=float(recipe["fused_query_weight"]),
                       n_params=sum(p.numel() for p in model.parameters()), checkpoint_mb=Path(path).stat().st_size / 1e6)


def design_library(data, design):
    """Reference median, its SIFT features, validity and zone masks for one design (loaded once, outside timing)."""
    key = design.replace("/", "_")
    median = cv2.imread(str(data / "processed/references" / f"{key}_all.png"), cv2.IMREAD_COLOR)
    coverage = cv2.imread(str(data / "processed/references" / f"{key}_it0_fold0_coverage.png"), cv2.IMREAD_GRAYSCALE)
    zone = cv2.imread(str(data / "processed/zones" / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
    assert median is not None and coverage is not None and zone is not None, design
    valid = ((coverage > 0).astype(np.uint8) * 255)
    return dict(median=median, feats=_features(median), masks=dict(target_valid=valid, valid=valid, zone=zone))


def stats(ms):
    s = sorted(ms)
    return dict(median_ms=round(statistics.median(s), 2), p95_ms=round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], 2),
                mean_ms=round(statistics.fmean(s), 2), n=len(s))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True); ap.add_argument("--on", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data")); ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None); ap.add_argument("--label", default=None, help="machine label for the record")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda: the forward passes run on this device, preprocessing stays on the CPU")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads); cv2.setNumThreads(a.threads); torch.set_grad_enabled(False)
    device = torch.device(a.device)
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    data = Path(a.data_root)
    split = pd.read_csv(a.split or data / "processed/splits/protocol_a_it0.csv", dtype=str)
    records = pd.read_csv(data / "processed/records.csv", dtype=str).set_index("image_id")
    docs = split[split.role == "test"].sort_values("image_id").head(a.n + a.warmup)
    models, info = {}, {}
    for arm in ("off", "on"):
        models[arm], info[arm] = load_arm(getattr(a, arm), a.weights, arm)
        models[arm].to(device)
    library = {d: design_library(data, d) for d in sorted(set(docs.design_id))}
    t = {k: [] for k in ("decode", "canvas", "registration", "reference", "tensor", "forward_off", "forward_on", "e2e_off", "e2e_on")}
    failed, unavailable = 0, 0
    for i, row in enumerate(docs.itertuples()):
        lib = library[row.design_id]; path = data / "raw" / records.at[row.image_id, "path"]
        t0 = time.perf_counter(); raw = cv2.imread(str(path), cv2.IMREAD_COLOR); t1 = time.perf_counter()
        canvas, qvalid, _ = to_canvas(raw); t2 = time.perf_counter()
        matrix, inliers = _fit(_features(canvas), lib["feats"], 12345)
        overlap, zone = np.zeros((512, 800), np.uint8), np.zeros((512, 800), np.uint8); inverse = None
        if matrix is not None:
            inverse, overlap = _overlap(qvalid, lib["masks"], _similarity(matrix))
            zone = (_warp(lib["masks"]["zone"], inverse, True) > 0).astype(np.uint8) * overlap
        else:
            failed += 1
        available = _available(overlap)
        if not available:
            overlap.fill(0); zone.fill(0); unavailable += 1
        t3 = time.perf_counter()
        reference = _warp(lib["median"], inverse) if (available and inverse is not None) else np.zeros((512, 800, 3), np.uint8)
        t4 = time.perf_counter()
        q = torch.from_numpy(np.ascontiguousarray(np.concatenate([_normalize(canvas), zone[None].astype(np.float32)])))[None]
        ov = torch.from_numpy(overlap[None].astype(np.float32))[None]
        avail = torch.tensor([available], dtype=torch.bool)
        ref = torch.from_numpy(np.ascontiguousarray(_normalize(reference)))[None]
        q, ov, avail, ref = q.to(device), ov.to(device), avail.to(device), ref.to(device); sync()
        t5 = time.perf_counter()
        models["off"](q, overlap=ov, reference=None, context_available=avail); sync(); t6 = time.perf_counter()
        models["on"](q, overlap=ov, reference=ref, context_available=avail); sync(); t7 = time.perf_counter()
        if i < a.warmup:
            continue
        ms = lambda x, y: 1e3 * (y - x)  # noqa: E731
        t["decode"].append(ms(t0, t1)); t["canvas"].append(ms(t1, t2)); t["registration"].append(ms(t2, t3))
        t["reference"].append(ms(t3, t4)); t["tensor"].append(ms(t4, t5)); t["forward_off"].append(ms(t5, t6)); t["forward_on"].append(ms(t6, t7))
        t["e2e_off"].append(ms(t0, t3) + ms(t4, t6)); t["e2e_on"].append(ms(t0, t5) + ms(t6, t7))
    host = platform.node().split(".")[0]
    summary = {k: stats(v) for k, v in t.items()}
    summary["documents_per_second"] = {arm: round(1e3 / summary[f"e2e_{arm}"]["median_ms"], 2) for arm in ("off", "on")}
    record = dict(machine=a.label or host, hostname=host, cpu=cpu_name(), threads=a.threads, logical_cpus=os.cpu_count(),
                  platform=platform.platform(), python=platform.python_version(), torch=torch.__version__, opencv=cv2.__version__,
                  batch_size=1, dtype="float32", device=a.device, gpu=(torch.cuda.get_device_name(0) if device.type == "cuda" else None), n=a.n, warmup=a.warmup, population="SIDTD fold-0 test files, first n by image_id",
                  designs=sorted(library), registration_failed=failed, context_unavailable=unavailable,
                  checkpoints={arm: dict(path=str(getattr(a, arm)), **info[arm]) for arm in ("off", "on")}, timing=summary)
    out = Path(a.out or ROOT / "results/timing" / f"deploy-{host}-{a.device}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(cpu=record["cpu"], threads=a.threads, n=a.n, **{k: summary[k]["median_ms"] for k in t}, fps=summary["documents_per_second"]), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
