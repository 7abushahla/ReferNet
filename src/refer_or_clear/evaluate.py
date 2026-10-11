"""Evaluate a model on the test variants and subsets of a protocol (implementation plan Section 5).

Scores are exported as bona fide probabilities, score = 1 - sigmoid(logit) (1 = genuine), and the
metrics follow metrics.py. The transported policy thresholds come from the source validation split
(bob far_threshold at 5 and 1 percent miss), frozen, then applied to every test variant and subset.
`run()` is called by train.py at the end of training; the CLI evaluates a saved checkpoint alone:

    python -m refer_or_clear.evaluate --checkpoint checkpoints/<tag>/best.pt --config configs/protocol_a.yaml \
        --split test --run-dir results/runs/<tag> [--data-root data/processed --device cpu --limit K]

Data plumbing shared with train.py (config, split csv, pairs per role, deterministic subsampling, loaders)
lives here so that both entry points build identical inputs.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .logging_utils import dump_json

REF_ALIAS = {"design_median": "median", "masked_exemplar": "specimen"}
APCER_TARGETS = (0.01, 0.05, 0.10)


# ----------------------------------------------------------------------------- data plumbing
def load_config(path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def reference_name(cfg: dict, override=None) -> str:
    r = override or cfg.get("data", {}).get("reference", "median")
    return REF_ALIAS.get(r, r)


def load_split(cfg: dict, data_root, config_path=None, fold=None):
    """(split_df, records_df): the protocol csv named in the config, resolved under data_root, and records.csv.
    A split name with a '{fold}' placeholder (Protocols A and B) needs `fold`."""
    import pandas as pd
    root = Path(data_root)
    name = Path(cfg["data"]["split"]).name
    if "{fold}" in name:
        if fold is None:
            raise ValueError(f"split {name} needs a fold (--fold K)")
        name = name.format(fold=int(fold))
    split = root / "splits" / name
    if not split.is_file():
        raise FileNotFoundError(f"split csv not found at {split}")
    split_df = pd.read_csv(split, dtype={"subset": str}, keep_default_na=False)
    records_df = pd.read_csv(root / "records.csv", keep_default_na=False)
    return split_df, records_df


def subsample(pairs: list[dict], limit, seed: int) -> list[dict]:
    """Deterministic class-balanced subsample of at most `limit` pairs (all of them when limit is None)."""
    if limit is None or len(pairs) <= limit:
        return pairs
    rng = np.random.RandomState(seed)
    alt = [i for i, p in enumerate(pairs) if p["label"] == 1]
    gen = [i for i, p in enumerate(pairs) if p["label"] == 0]
    k_alt = min(limit // 2, len(alt))
    k_gen = min(limit - k_alt, len(gen))
    k_alt = min(limit - k_gen, len(alt))
    keep = sorted(list(rng.choice(alt, k_alt, replace=False)) + list(rng.choice(gen, k_gen, replace=False)))
    return [pairs[i] for i in keep]


def role_pairs(split_df, records_df, data_root, role, reference, cfg, seed, limit=None,
               wrong_design_fraction=None) -> list[dict]:
    """Pairs of one role via pairs.build_pairs; wrong-design negatives only in training, at the given fraction
    (default: the config's); [] for an absent role."""
    from .data import pairs as P
    if not (split_df.role == role).any():
        return []
    if role != "train":
        frac = 0.0
    elif wrong_design_fraction is not None:
        frac = float(wrong_design_fraction)
    else:
        frac = float(cfg.get("data", {}).get("wrong_design_fraction", 0.1))
    out = P.build_pairs(split_df, records_df, data_root, reference=reference, wrong_design_fraction=frac,
                        role=role, seed=seed)
    return subsample(out, limit, seed)


def variant_kwargs(variant: str) -> dict:
    """Dataset keyword arguments of a test variant name: 'native' or 'equalized_qf<N>'."""
    if variant == "native":
        return {}
    if variant.startswith("equalized_qf"):
        return {"equalize_qf": int(variant[len("equalized_qf"):])}
    raise ValueError(f"unknown test variant {variant!r}")


def eval_loader(pairs, batch_size, workers, **kw):
    from .data.dataset import make_loader
    return make_loader(pairs, False, batch_size, workers, **kw)


# ----------------------------------------------------------------------------- inference
def _meta_list(meta, key, n):
    """Per-sample values of a collated meta field (list, tensor or missing)."""
    v = meta.get(key) if isinstance(meta, dict) else None
    if v is None:
        return [""] * n
    return [x.item() if torch.is_tensor(x) else x for x in v]


@torch.no_grad()
def predict(model, loader, device, amp=False, loss_fn=None, mask_weight=3.0, map_thr=0.5) -> dict:
    """Forward every batch once; per-sample bona fide score, label, map metrics and meta, plus the mean loss."""
    model.eval()
    use_amp = bool(amp) and device.type == "cuda"
    rec = {k: [] for k in ("score", "label", "loc_f1", "fp_in", "fp_out", "image_id", "subset", "wrong_design",
                           "attack_family", "design_id")}
    losses = []
    for batch in loader:
        x, mask, valid, label = (batch[k].to(device, non_blocking=True) for k in ("x", "mask", "valid", "label"))
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        logit, map_logit = out["logit"].float(), out["map"].float()
        if loss_fn is not None:
            losses.append(float(loss_fn(logit, map_logit, label, mask, valid, mask_weight)[0]))
        if map_logit.shape[-2:] != mask.shape[-2:]:
            map_logit = F.interpolate(map_logit, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        prob = torch.sigmoid(map_logit)[:, 0].cpu().numpy()
        gt, v = mask[:, 0].cpu().numpy() > 0.5, valid[:, 0].cpu().numpy() > 0.5
        zone = x[:, 3].float().cpu().numpy() > 0.5
        y = label.reshape(-1).cpu().numpy().astype(int)
        n = len(y)
        rec["score"].extend((1.0 - torch.sigmoid(logit.reshape(-1))).cpu().numpy().tolist())
        rec["label"].extend(y.tolist())
        rec["loc_f1"].extend(M.per_image_localization_f1(prob, gt, v, map_thr, labels=y).tolist())
        fi, fo = M.per_image_fp_area(prob, zone, v, map_thr)
        rec["fp_in"].extend(fi.tolist())
        rec["fp_out"].extend(fo.tolist())
        meta = batch.get("meta", {})
        rec["image_id"].extend(_meta_list(meta, "image_id", n))
        rec["subset"].extend(_meta_list(meta, "subset", n))
        rec["wrong_design"].extend(bool(w) for w in _meta_list(meta, "wrong_design", n))
        rec["attack_family"].extend(_meta_list(meta, "attack_family", n))
        rec["design_id"].extend(_meta_list(meta, "design_id", n))
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["loss"] = float(np.mean(losses)) if losses else None
    return out


def val_thresholds(pred: dict) -> dict:
    """Validation block: AUC and the frozen policy thresholds at 5 and 1 percent miss (None if undefined)."""
    s, y = pred["score"], pred["label"]
    return {"auc": M.auc(s, y) if len(y) else None,
            "threshold_miss5": M.far_threshold(s, y, 0.05) if len(y) else None,
            "threshold_miss1": M.far_threshold(s, y, 0.01) if len(y) else None,
            "n_genuine": int((y == 0).sum()), "n_altered": int((y == 1).sum())}


def metric_block(pred: dict, thresholds: dict | None, idx=None) -> dict:
    """The per-variant block of Section 5 on the samples selected by `idx`; None where a class is absent."""
    sel = np.arange(len(pred["label"])) if idx is None else np.asarray(idx)
    s, y = pred["score"][sel], pred["label"][sel]
    gen = sel[y == 0]
    thr = thresholds or {}
    t5, t1 = thr.get("threshold_miss5"), thr.get("threshold_miss1")
    both = bool((y == 0).any() and (y == 1).any())
    bp = M.bpcer_at_apcer(s, y, APCER_TARGETS) if both else {t: None for t in APCER_TARGETS}
    tr5, tr1 = M.transported(s, y, t5), M.transported(s, y, t1)
    # per-image areas were computed in predict(); the fp area is reported on genuine images only
    fp_in = float(pred["fp_in"][gen].mean()) if len(gen) else None
    fp_out = float(pred["fp_out"][gen].mean()) if len(gen) else None
    return {
        "auc": M.auc(s, y) if both else None,
        "eer": M.eer(s, y) if both else None,
        "deepid_acc": M.deepid_acc(s, y),
        "deepid_f1": M.deepid_f1(s, y) if both else None,
        "bpcer_at_apcer_1": bp[0.01], "bpcer_at_apcer_5": bp[0.05], "bpcer_at_apcer_10": bp[0.10],
        "transported_miss5": tr5["miss"], "transported_referral5": tr5["referral"],
        "transported_miss1": tr1["miss"], "transported_referral1": tr1["referral"],
        "hter5": M.hter(s, y, t5) if (both and t5 is not None) else None,
        "hter1": M.hter(s, y, t1) if (both and t1 is not None) else None,
        "loc_f1": float(pred["loc_f1"][sel].mean()) if len(sel) else None,
        "fp_area_in_zone": fp_in, "fp_area_out_zone": fp_out,
        "n_genuine": int((y == 0).sum()), "n_altered": int((y == 1).sum()),
    }


def run(model, loader_by_variant: dict, device, thresholds_from_val: dict | None, subsets_meta: dict | None = None,
        amp=False) -> dict:
    """{'test': {variant: block}, 'subsets': {subset: {variant: block}}} over every loader; empty subsets skipped.

    `subsets_meta` optionally maps image_id to subset and overrides the subset carried in the batch meta.
    """
    test, subsets = {}, {}
    for variant, loader in loader_by_variant.items():
        if loader is None or len(loader.dataset) == 0:
            continue
        pred = predict(model, loader, device, amp=amp)
        test[variant] = metric_block(pred, thresholds_from_val)
        subset_of = pred["subset"] if subsets_meta is None else \
            np.asarray([subsets_meta.get(i, "") for i in pred["image_id"]])
        for name, idx in derived_subsets(pred, subset_of).items():
            subsets.setdefault(name, {})[variant] = metric_block(pred, thresholds_from_val, idx)
    return {"test": test, "subsets": subsets}


def derived_subsets(pred: dict, subset_of) -> dict:
    """Index sets for reporting: each named subset; within each (or 'all' when unnamed) one set per attack
    family, genuine cards of that subset plus the altered cards of that family; and one set per design when
    the test spans several designs. Names: '<subset>', '<subset>:<family>', 'design:<design>'."""
    subset_of = np.asarray([str(s) for s in subset_of])
    label, fam, design = pred["label"], np.asarray([str(f) for f in pred["attack_family"]]), \
        np.asarray([str(d) for d in pred["design_id"]])
    out = {}
    names = sorted({s for s in subset_of if s}) or [""]
    for s in names:
        in_s = (subset_of == s) if s else np.ones(len(label), bool)
        if s:
            out[s] = np.flatnonzero(in_s)
        for f in sorted({x for x in fam[in_s & (label == 1)] if x}):
            idx = np.flatnonzero(in_s & ((label == 0) | (fam == f)))
            if (label[idx] == 0).any():
                out[f"{s or 'all'}:{f}"] = idx
    if len(set(design)) > 1:
        for d in sorted(set(design)):
            out[f"design:{d}"] = np.flatnonzero(design == d)
    return out


def summary_line(tag: str, val: dict | None, test: dict) -> str:
    def f(v):
        return "na" if v is None else f"{v:.4f}"
    parts = [f"[done] {tag}"]
    if val:
        parts.append(f"val_auc={f(val.get('auc'))}")
    for variant, b in test.items():
        parts.append(f"{variant}: auc={f(b['auc'])} eer={f(b['eer'])} miss5={f(b['transported_miss5'])} "
                     f"ref5={f(b['transported_referral5'])} loc_f1={f(b['loc_f1'])} "
                     f"n={b['n_genuine']}+{b['n_altered']}")
    return " | ".join(parts)


# ----------------------------------------------------------------------------- CLI
def checkpoint_fold(ck: dict, override=None):
    """The fold or group index to evaluate: the override when given, else the one the checkpoint trained on."""
    return override if override is not None else ck.get("config", {}).get("fold")


def configure_precision(config):
    if config.get("strict_fp32"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_checkpoint(path, device):
    """(model, checkpoint dict): rebuilds the model from the saved CLI args and loads the weights."""
    from .models import build_model
    ck = torch.load(path, map_location="cpu")
    configure_precision(ck["config"])
    a = ck["args"]
    model = build_model(arch=a["arch"], fusion=a["fusion"], init="scratch", in_channels=4,
                        bn_policy=ck["config"].get("bn_policy", "running"))
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", help="role to evaluate: test (default) or val")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--reference", default=None, help="median | specimen | single (default: checkpoint args)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="subsample K pairs per role")
    ap.add_argument("--out", default=None, help="output json (default <run-dir>/eval_<split>.json)")
    ap.add_argument("--fold", type=int, default=None,
                    help="fold or group index for Protocols A and B (default: checkpoint config)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    cfg = load_config(args.config)
    model, ck = load_checkpoint(args.checkpoint, device)
    if ck["config"].get("strict_fp32") and args.amp:
        raise ValueError("strict-FP32 checkpoints must not be evaluated with --amp")
    if ck["config"].get("bn_policy") == "batch_stats" and args.batch_size != 1:
        raise ValueError("batch_stats checkpoints must be evaluated one image at a time (--batch-size 1)")
    reference = args.reference or ck["args"].get("reference") or reference_name(cfg)
    seed = int(ck["args"].get("seed", 0))
    fold = checkpoint_fold(ck, args.fold)
    split_df, records_df = load_split(cfg, args.data_root, fold=fold)
    t0 = time.time()
    val_pairs = role_pairs(split_df, records_df, args.data_root, "val", reference, cfg, seed, args.limit)
    val = val_thresholds(predict(model, eval_loader(val_pairs, args.batch_size, args.workers), device, args.amp)) \
        if val_pairs else None
    pairs = role_pairs(split_df, records_df, args.data_root, args.split, reference, cfg, seed, args.limit)
    variants = cfg.get("eval", {}).get("test_variants", ["native", "equalized_qf75"])
    loaders = {v: eval_loader(pairs, args.batch_size, args.workers, **variant_kwargs(v)) for v in variants} \
        if pairs else {}
    res = run(model, loaders, device, val, amp=args.amp)
    out = {"tag": Path(args.run_dir).name, "protocol": cfg.get("protocol"), "split": args.split,
           "checkpoint": str(args.checkpoint), "epoch": ck.get("epoch"), "reference": reference, "fold": fold,
           "val": val, "test": res["test"], "subsets": res["subsets"], "seconds": time.time() - t0}
    out["precision"] = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                            cudnn_tf32=torch.backends.cudnn.allow_tf32, amp=args.amp)
    path = dump_json(args.out or Path(args.run_dir) / f"eval_{args.split}.json", out)
    print(summary_line(out["tag"], val, res["test"]))
    print(f"[evaluate] wrote {path}")


if __name__ == "__main__":
    main()
