#!/usr/bin/env python3
"""CPU inference timing for a trained checkpoint (E10).

Batch one, FP32, CPU, a fixed thread count. Per document, with time.perf_counter:
  preprocessing   decode the raw source image, letterbox to the canvas (rectify.to_canvas), register to the
                  design reference with SIFT + RANSAC (register.estimate_similarity, warp), build the
                  8-channel tensor with the zone channel; registration is also reported on its own
  network         forward of the 8-channel tensor (query plus reference)
  network cached  reference features cached per design. ChangerEx exchanges features between the two
                  branches, so a true cache is impossible; the single-branch cost (query backbone only,
                  reference halves taken from a cache) is reported as a lower bound and labelled as such.
                  For siamese_r18 the cache is exact (encoder(reference) stored, fusion recomputed).
  end to end      preprocessing + network of the same document
Median and p95 over --n documents (the first n of the split role after sorting by image_id) after
--warmup warm-ups, FPS from the median end-to-end and network times, parameter count and file size.
Design-level data (reference canvas, its SIFT features, the zone mask) is loaded once per design,
outside the timed region, as a deployed system would hold it. Quantization is out of scope.

    python scripts/bench_cpu.py --checkpoint checkpoints/<tag>/best.pt [--n 200 --warmup 20 --threads 4 --onnx]

Writes results/timing/<tag>-<host>.json. Raw images come from --raw-root via records.csv; when they are
absent the processed canvases stand in ("input": "canvas" in the JSON).
"""
from __future__ import annotations

import argparse
import copy
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear import evaluate as E  # noqa: E402
from refer_or_clear.data.dataset import _normalize  # noqa: E402
from refer_or_clear.data.rectify import to_canvas  # noqa: E402
from refer_or_clear.data.register import MIN_INLIERS, _features, estimate_similarity, warp  # noqa: E402
from refer_or_clear.logging_utils import dump_json, git_hash  # noqa: E402

CACHE_NOTE = {
    "changerex_r18": "not applicable to ChangerEx (feature exchange couples the branches); "
                     "single-branch cost reported as lower bound",
    "siamese_r18": "exact: encoder(reference) cached per design, fusion and heads recomputed",
}
PRE_KEYS = ("total", "decode", "letterbox", "registration", "tensor")


# ----------------------------------------------------------------------------- environment
def cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()),
            "min": float(a.min()), "max": float(a.max())}


def default_config(ck: dict) -> Path:
    """The config path stored in the checkpoint, else configs/protocol_<x>.yaml of its protocol."""
    p = Path(ck.get("args", {}).get("config", ""))
    if p.is_file():
        return p
    proto = str(ck.get("config", {}).get("protocol", "a")).lower()
    return ROOT / "configs" / f"protocol_{proto}.yaml"


# ----------------------------------------------------------------------------- per-design cache
class DesignCache:
    """Reference canvas, its SIFT features and the zone mask, keyed by file path; loaded once, untimed."""

    def __init__(self):
        self.ref, self.zone = {}, {}

    def reference(self, path):
        if path not in self.ref:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            self.ref[path] = (img, _features(img), _normalize(img))
        return self.ref[path]

    def zone_mask(self, path):
        if path not in self.zone:
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.zone[path] = (m > 127).astype(np.float32)[None]
        return self.zone[path]


def preprocess(src: str, pair: dict, cache: DesignCache):
    """(x (1, 8, H, W), stage times in ms, applied, inliers): decode, letterbox, register, tensor."""
    t0 = time.perf_counter()
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    canvas, _, _ = to_canvas(img)
    t2 = time.perf_counter()
    _, ref_feats, ref_norm = cache.reference(pair["ref_path"])
    R, n = estimate_similarity(_features(canvas), ref_feats)
    applied = R is not None and n >= MIN_INLIERS
    if applied:
        canvas = warp(canvas, R)
    t3 = time.perf_counter()
    z = cache.zone_mask(pair["zone_path"])
    x = torch.from_numpy(np.ascontiguousarray(np.concatenate([_normalize(canvas), z, ref_norm, z], 0)))[None]
    t4 = time.perf_counter()
    ms = {"total": t4 - t0, "decode": t1 - t0, "letterbox": t2 - t1, "registration": t3 - t2, "tensor": t4 - t3}
    return x, {k: 1e3 * v for k, v in ms.items()}, bool(applied), int(n)


# ----------------------------------------------------------------------------- cached-reference path
class SingleBranch:
    """Network cost with the reference branch served from a cache: exact for siamese_r18, a lower bound
    for ChangerEx (the cached halves are wrong in value because of the exchange layers, right in cost)."""

    def __init__(self, model, arch: str):
        self.model, self.arch, self.cache = model, arch, {}

    def available(self) -> bool:
        if self.arch == "siamese_r18":
            return self.model.fusion_kind in ("difference", "attention")
        return self.arch == "changerex_r18"

    def ref_feats(self, key, x):
        """Per-design reference features, computed once from the first document of that design."""
        if key not in self.cache:
            c = self.model.in_channels
            if self.arch == "siamese_r18":
                self.cache[key] = self.model.encoder(x[:, c:2 * c])
            else:
                _, ref = self.model.split(x)
                self.cache[key] = [torch.chunk(f, 2, dim=1)[1] for f in self.model.backbone(ref, ref)]
        return self.cache[key]

    def __call__(self, x, f_r):
        m, c = self.model, self.model.in_channels
        if self.arch == "siamese_r18":
            fused = m.fusion(m.encoder(x[:, :c]), f_r)
            return m.referral_head(fused[-1], m.map_head(fused))
        bb = m.backbone
        h = bb.maxpool(bb.stem(x[:, :c]) if bb.deep_stem else bb.relu(bb.norm1(bb.conv1(x[:, :c]))))
        feats = []
        for i, name in enumerate(bb.res_layers):
            h = getattr(bb, name)(h)
            feats.append(torch.cat([h, f_r[i]], dim=1))
        if m.attention is not None:
            for i, key in ((2, "stage16"), (3, "stage32")):
                f_q, f_rr = torch.chunk(feats[i], 2, dim=1)
                feats[i] = torch.cat([m.attention[key](f_q, f_rr), f_rr], dim=1)
        feat, mp = m.decode_head(feats)
        g = m.referral.descriptor(feat, mp)
        if m.attention is not None:
            g = m.attention["global"](g, feat)
        return m.referral.classify(g)


# ----------------------------------------------------------------------------- onnx
class _Tuple(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out["logit"], out["map"]


class _SpatialExchange(torch.nn.Module):
    """Open-CD SpatialExchange without masked assignment (every p-th column swapped)."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        m = (torch.arange(x1.shape[-1]) % self.p == 0).view(1, 1, 1, -1)
        return torch.where(m, x2, x1), torch.where(m, x1, x2)


class _ChannelExchange(_SpatialExchange):
    """Open-CD ChannelExchange without masked assignment (every p-th channel swapped)."""

    def forward(self, x1, x2):
        m = (torch.arange(x1.shape[1]) % self.p == 0).view(1, -1, 1, 1)
        return torch.where(m, x2, x1), torch.where(m, x1, x2)


def _export_copy(model):
    """Deep copy for export: InstanceNorm2d becomes the equal GroupNorm(C, C) (the exporter rejects
    instance_norm when the traced channel size is unknown, as in FDAF) and the Open-CD exchange layers
    become torch.where forms (their masked assignments export to an invalid graph). Same arithmetic."""
    try:
        from opencd.models.utils.interaction_layer import ChannelExchange, SpatialExchange
    except ImportError:
        ChannelExchange = SpatialExchange = ()
    m = copy.deepcopy(model)
    for mod in list(m.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.InstanceNorm2d):
                gn = torch.nn.GroupNorm(child.num_features, child.num_features, eps=child.eps, affine=child.affine)
                if child.affine:
                    gn.weight.data.copy_(child.weight.data), gn.bias.data.copy_(child.bias.data)
                setattr(mod, name, gn)
            elif isinstance(child, SpatialExchange):
                setattr(mod, name, _SpatialExchange(child.p))
            elif isinstance(child, ChannelExchange):
                setattr(mod, name, _ChannelExchange(child.p))
    return m


def onnx_session(model, x, threads: int, path: Path):
    """(session, info) or (None, info with the reason) when onnxruntime is missing or the export fails."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None, {"skipped": "onnxruntime not installed (pip install onnxruntime)"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(_Tuple(_export_copy(model)), x, str(path), input_names=["x"],
                              output_names=["logit", "map"], opset_version=17, do_constant_folding=True)
    except Exception as e:  # noqa: BLE001
        return None, {"skipped": f"onnx export failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}"}
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    ref = float(model(x)["logit"].reshape(-1)[0])
    got = float(sess.run(["logit"], {"x": x.numpy()})[0].reshape(-1)[0])
    return sess, {"onnxruntime_version": ort.__version__, "opset": 17, "file_mb": path.stat().st_size / 1e6,
                  "logit_abs_diff_first_doc": abs(ref - got)}


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="protocol yaml (default: the one stored in the checkpoint)")
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--raw-root", default="data/raw", help="raw images (records.csv paths); canvases if absent")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--split", default="test", help="role to draw documents from: test (default) or val")
    ap.add_argument("--reference", default=None, help="median | specimen | single (default: checkpoint args)")
    ap.add_argument("--out", default=None, help="output json (default results/timing/<tag>-<host>.json)")
    ap.add_argument("--onnx", action="store_true", help="export to ONNX and time with onnxruntime as well")
    ap.add_argument("--fold", type=int, default=None, help="fold or group index (default: the checkpoint's)")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.set_grad_enabled(False)
    model, ck = E.load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval().float()
    arch, fusion = ck["args"]["arch"], ck["args"]["fusion"]
    tag, host = Path(args.checkpoint).resolve().parent.name, platform.node().split(".")[0]
    cfg = E.load_config(args.config or default_config(ck))
    reference = args.reference or ck["args"].get("reference") or E.reference_name(cfg)
    seed = int(ck["args"].get("seed", 0))

    split_df, records_df = E.load_split(cfg, args.data_root, fold=E.checkpoint_fold(ck, args.fold))
    pairs = E.role_pairs(split_df, records_df, args.data_root, args.split, reference, cfg, seed)  # sorted by image_id
    if not pairs:
        raise SystemExit(f"[bench] no pairs in role {args.split!r}")
    docs = pairs[:args.n]
    rec = records_df.set_index("image_id")
    raw = [str(Path(args.raw_root) / rec.at[p["image_id"], "path"]) for p in docs]
    missing = sum(not Path(r).is_file() for r in raw)
    if missing:
        print(f"[bench] {missing}/{len(raw)} raw images missing under {args.raw_root}; timing the processed "
              f"canvases as input (letterbox is then a resize to the same size)")
    srcs, input_kind = (raw, "raw") if not missing else ([p["path"] for p in docs], "canvas")

    cache, single = DesignCache(), SingleBranch(model, arch)
    use_single = single.available()
    x0 = preprocess(srcs[0], docs[0], cache)[0]
    sess, onnx_info, onnx_dir = None, None, None
    if args.onnx:
        onnx_dir = tempfile.TemporaryDirectory()
        sess, onnx_info = onnx_session(model, x0, args.threads, Path(onnx_dir.name) / f"{tag}.onnx")
        if sess is None:
            print(f"[bench] onnx: {onnx_info['skipped']}")

    def run_one(src, pair):
        x, pre, applied, inl = preprocess(src, pair, cache)
        t0 = time.perf_counter()
        float(model(x)["logit"].reshape(-1)[0])
        r = {"pre": pre, "net": 1e3 * (time.perf_counter() - t0), "applied": applied, "inliers": inl}
        if use_single:
            f_r = single.ref_feats(pair["ref_path"], x)
            t1 = time.perf_counter()
            float(single(x, f_r).reshape(-1)[0])
            r["single"] = 1e3 * (time.perf_counter() - t1)
        if sess is not None:
            xn = x.numpy()
            t2 = time.perf_counter()
            sess.run(["logit"], {"x": xn})
            r["onnx"] = 1e3 * (time.perf_counter() - t2)
        return r

    print(f"[bench] {tag}: {arch}/{fusion}, reference {reference}, {len(docs)} documents of role {args.split}, "
          f"input {input_kind}, {args.threads} threads, warmup {args.warmup}")
    for i in range(args.warmup):
        run_one(srcs[i % len(srcs)], docs[i % len(docs)])
    t_start = time.time()
    rows = [run_one(s, p) for s, p in zip(srcs, docs)]
    wall = time.time() - t_start

    net = [r["net"] for r in rows]
    e2e = [r["pre"]["total"] + r["net"] for r in rows]
    timing = {
        "preprocessing_ms": {k: stats([r["pre"][k] for r in rows]) for k in PRE_KEYS},
        "network_ms": {"full": stats(net),
                       "cached_reference": stats([r["single"] for r in rows]) if use_single else None},
        "end_to_end_ms": stats(e2e),
        "fps": {"end_to_end": 1e3 / float(np.median(e2e)), "network": 1e3 / float(np.median(net))},
    }
    if sess is not None:
        timing["onnx_network_ms"] = stats([r["onnx"] for r in rows])
        timing["fps"]["onnx_network"] = 1e3 / timing["onnx_network_ms"]["median"]
    out = {
        "tag": tag, "checkpoint": str(args.checkpoint), "arch": arch, "fusion": fusion, "reference": reference,
        "protocol": cfg.get("protocol"), "split": args.split, "input": input_kind,
        "n": len(rows), "warmup": args.warmup, "threads": args.threads, "batch_size": 1, "dtype": "float32",
        "device": "cpu", "hostname": platform.node(), "cpu": cpu_model(), "cpu_count": os.cpu_count(),
        "platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
        "opencv": cv2.__version__, "git_hash": git_hash(ROOT),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_mb": Path(args.checkpoint).stat().st_size / 1e6,
        "input_shape": list(x0.shape),
        "reference_cache": CACHE_NOTE[arch] if use_single else
                           f"not applicable to {arch}/{fusion} (no reference branch in the forward)",
        "design_cache": "reference canvas, its SIFT features and the zone mask are loaded once per design "
                        "outside the timed region",
        "registration": {"applied": int(sum(r["applied"] for r in rows)),
                         "mean_inliers": float(np.mean([r["inliers"] for r in rows])), "min_inliers": MIN_INLIERS},
        "timing": timing, "onnx": onnx_info, "wall_seconds": wall,
        "per_document_ms": {"preprocessing": [r["pre"]["total"] for r in rows], "network": net},
    }
    path = dump_json(args.out or ROOT / "results" / "timing" / f"{tag}-{host}.json", out)
    if onnx_dir is not None:
        onnx_dir.cleanup()

    def row(name, s):
        return f"  {name:<28s} {s['median']:9.1f} {s['p95']:9.1f}"
    print(f"\n{tag} on {out['cpu']} ({args.threads} threads, torch {torch.__version__}, input {input_kind})")
    print(f"  {'stage (ms)':<28s} {'median':>9s} {'p95':>9s}")
    for k in PRE_KEYS:
        print(row("preprocessing" if k == "total" else f"  {k}", timing["preprocessing_ms"][k]))
    print(row("network (query+reference)", timing["network_ms"]["full"]))
    if use_single:
        print(row("network (reference cached)", timing["network_ms"]["cached_reference"]))
    if sess is not None:
        print(row("network (onnxruntime)", timing["onnx_network_ms"]))
    print(row("end to end", timing["end_to_end_ms"]))
    print(f"  fps: end to end {timing['fps']['end_to_end']:.2f}, network {timing['fps']['network']:.2f}; "
          f"params {out['n_params'] / 1e6:.2f}M, checkpoint {out['checkpoint_mb']:.1f} MB; "
          f"registration applied {out['registration']['applied']}/{len(rows)}")
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    main()
