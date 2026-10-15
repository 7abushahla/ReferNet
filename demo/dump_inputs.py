"""Dump the exact network inputs of the demo samples from the artifacts behind the paper's results (run on the cluster).

SIDTD: mirrors ProtocolAReferenceDataset.__getitem__ of the frozen study source (study-v2-fdb7351) on the fold-0
evaluation/registration.json entries, with the q75 treatment of evaluate_reencoded (query re-encoded after decode,
before the canvas). IDNet: mirrors c_robust.Entries (quality=75, align=False) on each design's queries.json and
bank/, as diag-align/c_idnet.py scored them. Each sample is written as lossless PNGs, rescored in PyTorch from those
PNGs with the fold-0 checkpoints, and every logit must match the paper's locked predictions within 2e-3, or the
dump aborts.

Usage (GPU node, frozen source and diag-align on PYTHONPATH via sources.json): python demo/dump_inputs.py --out demo/out
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "demo/sources.json"
WEIGHTS_SHA256 = "da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618"
H, W = 512, 800


def write_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def write_masks(path: Path, zone: np.ndarray, overlap: np.ndarray):
    """One RGB PNG: R = zone, G = overlap (0 or 255), B = 0."""
    m = np.zeros((*zone.shape, 3), np.uint8)
    m[..., 0], m[..., 1] = (zone > 0) * 255, (overlap > 0) * 255
    write_rgb(path, m)


def read_masks(path: Path):
    m = read_rgb(path)
    return (m[..., 0] > 127).astype(np.uint8), (m[..., 1] > 127).astype(np.uint8)


def _path(value):
    return value["path"] if isinstance(value, dict) else value


def _reencode(bgr, quality):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _finish(query, valid, overlap, zone, reference, available_fn):
    available = available_fn(overlap)
    if not available:
        zone.fill(0); overlap.fill(0)
    zone *= overlap
    return dict(query=query, valid=valid, overlap=overlap, zone=zone, reference=reference, available=available)


def sidtd_arrays(entry: dict, raw_root: Path, quality: int) -> dict:
    """ProtocolAReferenceDataset.__getitem__ (reference enabled, no augmentation) with the q75 query treatment."""
    from refer_or_clear.data.direct_reference import _available, _overlap, _warp
    from refer_or_clear.data.rectify import to_canvas
    p = entry["products"]
    masks = {n: cv2.imread(_path(p[n]), cv2.IMREAD_GRAYSCALE) for n in ("target_valid", "valid", "zone")}
    query, valid, _ = to_canvas(_reencode(cv2.imread(str(raw_root / entry["path"]), cv2.IMREAD_COLOR), quality))
    overlap, zone, reference = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8), None
    if entry["status"] == "fit":
        inverse, overlap = _overlap(valid, masks, np.asarray(entry["query_canvas_to_bank"]))
        zone = (_warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap
        if _available(overlap):
            reference = _warp(cv2.imread(_path(p["median"]), cv2.IMREAD_COLOR), inverse)
    return _finish(query, valid, overlap, zone, reference, _available)


def idnet_arrays(entry: dict, design_root: Path, bank: dict, quality: int) -> dict:
    """c_robust.Entries.__getitem__ (reference enabled, train=False, align=False) with quality=75."""
    import c_robust as cr
    from refer_or_clear.data.direct_reference import _available
    from refer_or_clear.data.rectify import to_canvas
    masks = {n: cr.read_mask(bank[n]) for n in ("target_valid", "valid", "zone")}
    query, valid, _ = to_canvas(cr.reencode(cv2.imread(str(design_root / entry["path"]), cv2.IMREAD_COLOR), quality))
    overlap, zone, reference = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8), None
    if entry["matrix"] is not None:
        inverse, overlap = cr.overlap_of(valid, masks, np.asarray(entry["matrix"]))
        zone = (cr.warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap
        if _available(overlap):
            reference = cr.warp(cv2.imread(bank["median"], cv2.IMREAD_COLOR), inverse)
    return _finish(query, valid, overlap, zone, reference, _available)


def write_sample(d: Path, arrays: dict):
    assert arrays["available"] and arrays["reference"] is not None, d.name
    write_rgb(d / "query.png", cv2.cvtColor(arrays["query"], cv2.COLOR_BGR2RGB))
    write_rgb(d / "reference.png", cv2.cvtColor(arrays["reference"], cv2.COLOR_BGR2RGB))
    write_masks(d / "masks.png", arrays["zone"], arrays["overlap"])


def load_models(src: dict, device):
    import torch
    import c_robust as cr
    from refer_or_clear.models.direct_reference import build_direct_reference_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {}
    for arm in ("off", "on"):
        on = arm == "on"
        m = build_direct_reference_model(Path(src["a"]["weights"]), WEIGHTS_SHA256, reference_enabled=on, head_seed=42,
                                         adapter_seed=42, local_enabled=on, reference_bn="shared").to(device)
        cr.load_state(m, str(Path(src["a"]["runs"]) / arm / "best.pt"))
        models[arm] = m.eval()
    return models


def rescore(d: Path, models: dict, device, cudnn_tf32: bool = False, write: bool = True) -> dict:
    """Score the written PNGs (not the in-memory arrays), so the check covers exactly what the site ships.

    cudnn_tf32 reproduces a scorer's convolution precision: the SIDTD evaluation ran strict FP32
    (pilots.configure_precision), the IDNet scoring (c_idnet.py) ran PyTorch's default cuDNN TF32. The scores written
    for the site (write=True) are always strict FP32, the precision of the browser runtime."""
    import torch
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    from refer_or_clear.data.dataset import _normalize
    bgr = lambda name: cv2.cvtColor(read_rgb(d / name), cv2.COLOR_RGB2BGR)  # noqa: E731
    zone, overlap = read_masks(d / "masks.png")
    x = torch.from_numpy(np.concatenate([_normalize(bgr("query.png")), zone[None].astype(np.float32)]))[None].to(device)
    ov = torch.from_numpy(overlap[None, None].astype(np.float32)).to(device)
    ref = torch.from_numpy(np.ascontiguousarray(_normalize(bgr("reference.png"))))[None].to(device)
    avail = torch.tensor([True], device=device)
    with torch.inference_mode():
        on = models["on"](x, overlap=ov, reference=ref, context_available=avail)
        off = models["off"](x, overlap=ov, reference=None, context_available=avail)
    s = {"on": {"query_logit": float(on["query_logit"][0]), "correction_logit": float(on["correction_logit"][0])},
         "off": {"query_logit": float(off["attack_logit"][0])}}
    torch.backends.cudnn.allow_tf32 = False
    if write:
        (d / "scores.json").write_text(json.dumps(s))
    return s


def check(sample: dict, s: dict, tol: float = 2e-3):
    p = sample["paper"]
    errs = {"on_query": abs(s["on"]["query_logit"] - p["on"]["query_logit"]),
            "on_correction": abs(s["on"]["correction_logit"] - p["on"]["correction_logit"]),
            "off": abs(s["off"]["query_logit"] - p["off"]["attack_logit"])}
    bad = {k: round(v, 6) for k, v in errs.items() if v > tol}
    if bad:
        raise SystemExit(f"{sample['sample_id']} differs from the paper: {bad}")
    return max(errs.values())


def design_images(d: Path, bank: dict):
    write_rgb(d / "reference.png", cv2.cvtColor(cv2.imread(bank["median"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    zone = cv2.imread(bank["zone"], cv2.IMREAD_GRAYSCALE)
    write_masks(d / "zone.png", zone, np.zeros_like(zone))


def fits_or_false(attempt) -> bool:
    """Run a registration attempt; errors count as 'registration failed', as in the paper's registration
    (protocol_a_reference catches ValueError, cv2.error and LinAlgError)."""
    try:
        return bool(attempt())
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return False


def registers(arrays: dict, bank: dict) -> bool:
    """The paper's similarity fit of a query canvas to another design's bank, and a usable overlap."""
    import c_robust as cr
    from refer_or_clear.data.direct_reference import _available
    from refer_or_clear.data.register import _features
    masks = {n: cr.read_mask(bank[n]) for n in ("target_valid", "valid", "zone")}

    def attempt():
        M, _ = cr.fit(_features(arrays["query"]), _features(cv2.imread(bank["target"], cv2.IMREAD_COLOR)), "similarity", cr.SEED)
        return M is not None and _available(cr.overlap_of(arrays["valid"], masks, M)[1])
    return fits_or_false(attempt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo/out"))
    a = ap.parse_args(argv)
    src = json.loads(SOURCES.read_text())
    sys.path[:0] = [str(Path(src["a"]["source"]) / "src"), str(Path(src["a"]["source"]) / "scripts"),
                    src["a"]["diag_align"], str(ROOT / "demo")]
    import torch
    from select_samples import select_mismatch, select_samples
    out, q = Path(a.out), int(src["quality"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(src, device)
    samples = select_samples()
    registration = json.loads(Path(src["a"]["registration"]).read_text())["entries"]
    queries = {code: {e["image_id"]: e for e in json.loads((Path(v["run_dir"]) / "queries.json").read_text())}
               for code, v in src["c"].items()}
    banks, arrays_of, worst = {}, {}, 0.0
    for s in samples:
        if s["dataset"] == "sidtd":
            entry = registration[s["image_id"]]
            arrays = sidtd_arrays(entry, Path(src["a"]["raw_root"]), q)
            bank = {n: _path(entry["products"][n]) for n in ("target", "target_valid", "median", "valid", "zone")}
        else:
            code = s["design_id"].split("_", 1)[1]
            run = Path(src["c"][code]["run_dir"])
            bank = {n: str(run / "bank" / f"{n}.png") for n in ("target", "target_valid", "median", "valid", "zone")}
            arrays = idnet_arrays(queries[code][s["image_id"]], Path(src["c"][code]["design_root"]), bank, q)
        d = out / "samples" / s["sample_id"]
        write_sample(d, arrays)
        rescore(d, models, device)                                    # strict FP32 scores for the site
        worst = max(worst, check(s, rescore(d, models, device, cudnn_tf32=s["dataset"] == "idnet", write=False)))
        if s["design_id"] not in banks:
            banks[s["design_id"]] = bank
            design_images(out / "designs" / s["design_id"], bank)
        arrays_of[s["sample_id"]] = arrays
    print(f"{len(samples)} samples verified against the paper (max |logit diff| {worst:.2e}); {len(banks)} designs")
    mm = {}
    for s in select_mismatch(samples):
        mm[s["sample_id"]] = {d: ("fit" if registers(arrays_of[s["sample_id"]], b) else "no_fit")
                              for d, b in banks.items() if d != s["design_id"]}
    (out / "mismatch.json").write_text(json.dumps(mm, indent=1))
    print("mismatch fits:", {k: sum(v == "fit" for v in m.values()) for k, m in mm.items()})


if __name__ == "__main__":
    main()
