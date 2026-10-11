"""Train one configuration and evaluate it (implementation plan Section 5). Called by the sweep jobs:

    python -m refer_or_clear.train --config configs/protocol_a.yaml --arch siamese_r18 --fusion difference \
        --reference median --init imagenet --seed 42 --data-root data/processed --run-dir results/runs/<tag> \
        [--epochs N --limit K --device cpu|cuda --amp --batch-size B --workers W --crop H W]

AdamW with two groups (pretrained encoder at train.lr_backbone, new modules at train.lr_new), linear warm-up
then cosine to zero, AMP on CUDA, gradient clipping at 5. Every arm trains the full schedule; the checkpoint
chosen by train.select is evaluated: best validation accuracy with AUC breaking ties (Protocols A and B, the
SIDTD paper's rule) or best validation AUC (train.patience in the config can re-enable early stopping).
Writes checkpoints/<tag>/{best,last}.pt, <run-dir>/log.csv, <run-dir>/train.json, and LAST, only on success,
<run-dir>/metrics.json (the sweep skips arms whose metrics.json exists). Scores are bona fide probabilities.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import evaluate as E
from . import metrics as M
from .logging_utils import CsvLogger, dump_json, git_hash

LOG_COLUMNS = ["epoch", "train_loss", "bce_image", "bce_map", "dice_map", "val_loss", "val_acc", "val_auc", "lr", "seconds",
               "peak_mem_mb"]
PATIENCE = None   # no early stopping by default (decision of 10 Oct: identical compute for every arm)
GRAD_CLIP = 5.0
SINGLE_FUSIONS = ("single", "single_noex", "attention_single")   # ignore the reference: no wrong-design pairs for them


def jpeg_range(rc: dict):
    """(low, high) JPEG quality for training augmentation, or None when the config switches it off."""
    q = rc.get("augment", {}).get("jpeg_quality")
    return tuple(int(v) for v in q) if q else None


def selection_key(select: str, val_acc, val_auc, val_loss, train_loss) -> tuple:
    """Larger is better; ties keep the earlier epoch. 'accuracy': (validation accuracy, validation AUC), the SIDTD
    paper's rule with AUC breaking ties; 'auc': (validation AUC,). Without validation metrics: -val_loss, then
    -train_loss."""
    if select == "accuracy" and val_acc is not None:
        return (val_acc, val_auc if val_auc is not None else -math.inf)
    if val_auc is not None:
        return (val_auc,)
    return (-val_loss,) if val_loss is not None else (-train_loss,)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", default="siamese_r18")
    ap.add_argument("--fusion", default=None, help="single | stacked | difference | attention (default: config)")
    ap.add_argument("--reference", default=None, help="median | specimen | single (default: config)")
    ap.add_argument("--init", default=None, help="imagenet | scratch (default: config model.init)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=None, help="fold (Protocol A) or group (Protocol B) index")
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="default: config train.epochs")
    ap.add_argument("--limit", type=int, default=None, help="subsample K pairs per role (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: config train.batch_size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="training crop H W; default from augment.crop in the config (null = full canvas)")
    ap.add_argument("--bn-policy", choices=("running", "batch_stats"), default=None,
                    help="encoder BN statistics; batch_stats scores each validation/test image alone")
    ap.add_argument("--validation-only", action="store_true", help="development pilot: never construct or score test pairs")
    ap.add_argument("--strict-fp32", action="store_true", help="disable TF32 and reject AMP for controlled comparisons")
    return ap.parse_args(argv)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def peak_mem_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, kilobytes on Linux
    return rss / 2 ** 20 if sys.platform == "darwin" else rss / 2 ** 10


def resolve(args, cfg) -> dict:
    """The run configuration with CLI overrides applied; everything train.json records."""
    t, m = cfg.get("train", {}), cfg.get("model", {})
    if args.strict_fp32 and args.amp:
        raise ValueError("strict FP32 cannot be combined with AMP")
    init = args.init or m.get("init", "imagenet")
    allowed = ("scratch", "levir") if args.arch == "changerex_r18" else ("imagenet", "scratch")
    if init not in allowed:
        raise SystemExit(f"[train] --init {init!r} is not available for arch {args.arch}; use one of {allowed}")
    fusion = args.fusion or m.get("fusion", "difference")
    bn_policy = args.bn_policy or m.get("bn_policy", "running")
    if bn_policy == "batch_stats" and args.arch != "changerex_r18":
        raise SystemExit("[train] batch_stats BN policy requires changerex_r18")
    return dict(
        tag=Path(args.run_dir).name, protocol=cfg.get("protocol"), arch=args.arch,
        fusion=fusion, bn_policy=bn_policy, reference=E.reference_name(cfg, args.reference), fold=args.fold,
        validation_only=args.validation_only, strict_fp32=args.strict_fp32,
        select=str(t.get("select", "auc")),
        init=init, seed=args.seed, epochs=int(t.get("epochs", 40) if args.epochs is None else args.epochs),
        patience=t.get("patience"),
        batch_size=int(t.get("batch_size", 8) if args.batch_size is None else args.batch_size),
        eval_batch_size=1 if bn_policy == "batch_stats" else int(t.get("batch_size", 8) if args.batch_size is None else args.batch_size),
        lr_backbone=float(t.get("lr_backbone", 1e-4)), lr_new=float(t.get("lr_new", 1e-3)),
        weight_decay=float(t.get("weight_decay", 0.05)), warmup_iters=int(t.get("warmup_iters", 500)),
        mask_weight=float(t.get("mask_loss_weight", 3.0)), amp=bool(args.amp), workers=args.workers,
        crop=tuple(args.crop) if args.crop else (tuple(cfg.get("augment", {}).get("crop")) if cfg.get("augment", {}).get("crop") else None),
        limit=args.limit, data_root=str(args.data_root),
        wrong_design_fraction=0.0 if fusion in SINGLE_FUSIONS
        else float(cfg.get("data", {}).get("wrong_design_fraction", 0.1)),
        augment=dict(cfg.get("augment", {})),
        test_variants=list(cfg.get("eval", {}).get("test_variants", ["native", "equalized_qf75"])),
    )


def evaluation_roles(rc):
    return ("train", "val") if rc.get("validation_only") else ("train", "val", "test")


def make_optimizer(model, rc):
    groups = model.parameter_groups() if hasattr(model, "parameter_groups") else ([], list(model.parameters()))
    enc, new = groups
    lr_enc = rc["lr_backbone"] if rc["init"] != "scratch" else rc["lr_new"]  # nothing is pretrained at scratch
    params = [{"params": new, "lr": rc["lr_new"], "name": "new"}]
    if enc:
        params.insert(0, {"params": enc, "lr": lr_enc, "name": "encoder"})
    return torch.optim.AdamW(params, weight_decay=rc["weight_decay"])


def make_scheduler(opt, total_iters: int, warmup: int):
    warmup = min(warmup, int(0.1 * total_iters))

    def f(it):
        if it < warmup:
            return (it + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, total_iters - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_epoch(model, loader, opt, sched, scaler, device, loss_fn, rc):
    model.train()
    use_amp = rc["amp"] and device.type == "cuda"
    sums = {"loss": 0.0, "bce_image": 0.0, "bce_map": 0.0, "dice_map": 0.0}
    n = 0
    for batch in loader:
        x, mask, valid, label = (batch[k].to(device, non_blocking=True) for k in ("x", "mask", "valid", "label"))
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        loss, parts = loss_fn(out["logit"].float(), out["map"].float(), label, mask, valid, rc["mask_weight"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {loss.item()} at iteration {n}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        sched.step()
        sums["loss"] += loss.item()
        for k in ("bce_image", "bce_map", "dice_map"):
            sums[k] += parts[k]
        n += 1
    return {k: (v / n if n else float("nan")) for k, v in sums.items()}, n


def save_checkpoint(path, model, rc, args, epoch, val_auc, n_params):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": rc, "args": vars(args), "epoch": epoch,
                "val_auc": val_auc, "n_params": n_params}, path)


def main(argv=None):
    args = parse_args(argv)
    from .data.dataset import make_loader  # data and model modules are built in parallel; import late
    from .losses import referral_loss
    from .models import build_model, count_params

    cfg = E.load_config(args.config)
    rc = resolve(args, cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[train] --device cuda requested but CUDA is not available")
    seed_everything(rc["seed"])
    E.configure_precision(rc)
    run_dir, ck_dir = Path(args.run_dir), Path("checkpoints") / rc["tag"]
    run_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    split_df, records_df = E.load_split(cfg, args.data_root, fold=rc["fold"])
    pairs = {role: E.role_pairs(split_df, records_df, args.data_root, role, rc["reference"], cfg, rc["seed"],
                                args.limit, wrong_design_fraction=rc["wrong_design_fraction"])
             for role in evaluation_roles(rc)}
    if not pairs["train"]:
        raise SystemExit("[train] no training pairs in this split")
    for role in evaluation_roles(rc)[1:]:
        if not pairs[role]:
            print(f"[train] warning: no {role} pairs in this split")
    aug = rc["augment"]
    train_loader = make_loader(pairs["train"], True, rc["batch_size"], rc["workers"], crop=rc["crop"],
                               jpeg=jpeg_range(rc), affine_px=aug.get("affine_px", 8),
                               photometric=aug.get("photometric", True))
    val_loader = E.eval_loader(pairs["val"], rc["eval_batch_size"], rc["workers"]) if pairs["val"] else None

    model = build_model(arch=rc["arch"], fusion=rc["fusion"], init=rc["init"], in_channels=4,
                        bn_policy=rc["bn_policy"]).to(device)
    n_params = count_params(model)
    opt = make_optimizer(model, rc)
    total_iters = rc["epochs"] * len(train_loader)
    sched = make_scheduler(opt, max(1, total_iters), rc["warmup_iters"])
    scaler = torch.cuda.amp.GradScaler(enabled=rc["amp"] and device.type == "cuda")
    info = dict(config=rc, raw_config=cfg, args=vars(args), git_hash=git_hash(), n_params=n_params,
                device=str(device), torch_version=torch.__version__, python=sys.version.split()[0],
                n_pairs={k: len(v) for k, v in pairs.items()}, iters_per_epoch=len(train_loader),
                total_iters=total_iters, hostname=os.uname().nodename)
    info["precision"] = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                             cudnn_tf32=torch.backends.cudnn.allow_tf32, amp=rc["amp"])
    dump_json(run_dir / "train.json", info)
    print(f"[train] {rc['tag']}: {n_params / 1e6:.2f}M params on {device}, "
          f"{info['n_pairs']} pairs, {len(train_loader)} it/epoch x {rc['epochs']} epochs")

    log = CsvLogger(run_dir / "log.csv", LOG_COLUMNS)
    best_score, best_epoch, bad_epochs = None, 0, 0
    for epoch in range(1, rc["epochs"] + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tr, n_it = train_epoch(model, train_loader, opt, sched, scaler, device, referral_loss, rc)
        val_loss = val_auc = val_acc = None
        if val_loader is not None:
            vp = E.predict(model, val_loader, device, amp=rc["amp"], loss_fn=referral_loss,
                           mask_weight=rc["mask_weight"])
            val_loss, val_auc = vp["loss"], M.auc(vp["score"], vp["label"])
            val_acc = M.deepid_acc(vp["score"], vp["label"])
        lr = opt.param_groups[-1]["lr"]
        log.log(epoch=epoch, train_loss=tr["loss"], bce_image=tr["bce_image"], bce_map=tr["bce_map"],
                dice_map=tr["dice_map"], val_loss=val_loss, val_acc=val_acc, val_auc=val_auc, lr=lr, seconds=time.time() - t0,
                peak_mem_mb=peak_mem_mb(device))
        print(f"[epoch {epoch}/{rc['epochs']}] loss {tr['loss']:.4f} val_loss "
              f"{'na' if val_loss is None else f'{val_loss:.4f}'} val_auc "
              f"{'na' if val_auc is None else f'{val_auc:.4f}'} val_acc {'na' if val_acc is None else f'{val_acc:.4f}'} lr {lr:.2e} {time.time() - t0:.0f}s "
              f"{peak_mem_mb(device):.0f}MB")
        # selection: train.select (accuracy with AUC ties, or AUC); falls back to -val_loss, then -train loss
        score = selection_key(rc["select"], val_acc, val_auc, val_loss, tr["loss"])
        save_checkpoint(ck_dir / "last.pt", model, rc, args, epoch, val_auc, n_params)
        if best_score is None or score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(ck_dir / "best.pt", model, rc, args, epoch, val_auc, n_params)
        else:
            bad_epochs += 1
            patience = rc.get("patience") or PATIENCE
            if patience and val_auc is not None and bad_epochs >= patience:
                print(f"[train] early stop at epoch {epoch}; best epoch {best_epoch}")
                break
    train_seconds = time.time() - t_start

    # final evaluation with the best checkpoint; thresholds frozen on validation
    if (ck_dir / "best.pt").is_file():
        model.load_state_dict(torch.load(ck_dir / "best.pt", map_location="cpu")["state_dict"])
    val = None
    if val_loader is not None:
        val = E.val_thresholds(E.predict(model, val_loader, device, amp=rc["amp"]))
    loaders = {v: E.eval_loader(pairs["test"], rc["eval_batch_size"], rc["workers"], **E.variant_kwargs(v))
               for v in rc["test_variants"]} if pairs.get("test") else {}
    res = E.run(model, loaders, device, val, amp=rc["amp"])
    info.update(timing=dict(train_seconds=train_seconds, epochs=epoch if rc["epochs"] else 0,
                            best_epoch=best_epoch, total_seconds=time.time() - t_start))
    dump_json(run_dir / "train.json", info)
    metrics = {"tag": rc["tag"], "protocol": rc["protocol"], "arch": rc["arch"], "fusion": rc["fusion"],
               "bn_policy": rc["bn_policy"],
               "validation_only": rc["validation_only"], "strict_fp32": rc["strict_fp32"],
               "reference": rc["reference"], "init": rc["init"], "seed": rc["seed"], "fold": rc["fold"], "n_params": n_params,
               "val": val, "test": res["test"], "subsets": res["subsets"],
               "timing": {"train_seconds": train_seconds, "epochs": epoch if rc["epochs"] else 0,
                          "best_epoch": best_epoch}}
    dump_json(run_dir / "metrics.json", metrics)  # last, and only on success
    print(E.summary_line(rc["tag"], val, res["test"]) + f" | {train_seconds:.0f}s")


if __name__ == "__main__":
    main()
