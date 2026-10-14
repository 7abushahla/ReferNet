"""Train a published SIDTD baseline (ResNet50 or EfficientNet-B3, the two with released checkpoints) under Protocol B:
the six source nationalities of a group train it, their held-out identities validate it, and the frozen Protocol B query
population of the four held-out nationalities (identical to the OFF/ON queries) is scored, native and re-encoded at Q75.

Recipe copied from SIDTD_Dataset/models/Baseline (training_kfold_baseline.py, _utils.py): ImageNet-pretrained backbone,
two-class head, input 299 x 299, RandomResizedCrop(scale 0.8-1.0) + horizontal and vertical flip (p 0.5) +
RandomBrightnessContrast(p 0.2), ImageNet normalization, SGD lr 0.01 momentum 0.9, ReduceLROnPlateau(factor 0.9,
patience 1) on the validation loss, batch 32, 100 epochs, seed 777, checkpoint = best validation accuracy.  The raw
files are read as published (no canvas, no re-encoding), which is the published input path.

Usage (one job per group and model):
  python scripts/baselines/sidtd_baseline_protocol_b.py --run-dir results/runs/protocol-b-current/protocol-b-group1-38172
         --raw-root data/raw --model resnet50 --out results/baselines/protocol-b/group1/resnet50 [--epochs 100]
Outputs: best.pt, train_log.csv, {native,q75}_predictions.csv (image_id,label,family,attack_logit,p_altered),
summary.json (AUC, text-family AUCs, accuracy and genuine false positives at argmax, selected epoch).
"""
import argparse, csv, json, random, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 777; SIZE = 299
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(name):
    if name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1); m.avgpool = nn.AdaptiveAvgPool2d(1); m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet-b3":
        from efficientnet_pytorch import EfficientNet
        m = EfficientNet.from_pretrained("efficientnet-b3"); m._fc = nn.Linear(m._fc.in_features, 2)
    else:
        raise ValueError(name)
    return m


def reencode(bgr, q):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)]); assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def train_augment(rgb, rng):
    """albumentations RandomResizedCrop(299, 299, scale=(0.8, 1.0), ratio=(3/4, 4/3)) + flips + brightness/contrast."""
    h, w = rgb.shape[:2]; area = h * w
    for _ in range(10):
        scale = rng.uniform(0.8, 1.0); log_ratio = (np.log(3 / 4), np.log(4 / 3)); ratio = np.exp(rng.uniform(*log_ratio))
        cw = int(round(np.sqrt(area * scale * ratio))); ch = int(round(np.sqrt(area * scale / ratio)))
        if 0 < cw <= w and 0 < ch <= h:
            y0 = rng.randint(0, h - ch); x0 = rng.randint(0, w - cw); rgb = rgb[y0:y0 + ch, x0:x0 + cw]; break
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5: rgb = rgb[:, ::-1]
    if rng.random() < 0.5: rgb = rgb[::-1, :]
    if rng.random() < 0.2:  # RandomBrightnessContrast defaults: limits 0.2, contrast then brightness
        alpha = 1.0 + rng.uniform(-0.2, 0.2); beta = rng.uniform(-0.2, 0.2)
        rgb = np.clip(rgb.astype(np.float32) * alpha + beta * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class Imgs(Dataset):
    def __init__(self, rows, raw_root, train, quality=None, seed=SEED):
        self.rows, self.root, self.train, self.quality, self.seed = rows, Path(raw_root), train, quality, seed; self.epoch = 0

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; bgr = cv2.imread(str(self.root / r["path"]), cv2.IMREAD_COLOR); assert bgr is not None, r["path"]
        if self.quality: bgr = reencode(bgr, self.quality)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.train: rgb = train_augment(rgb, random.Random(self.seed * 1_000_003 + self.epoch * 100_003 + i))
        else: rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        return to_tensor(rgb), int(r["label"]), i


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n): return None
    return float((p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean())


@torch.no_grad()
def score(model, loader, dev):
    model.eval(); out = {}
    for x, y, idx in loader:
        logits = model(x.to(dev, non_blocking=True)).float().cpu(); lp = torch.log_softmax(logits, 1)
        for k, i in enumerate(idx.tolist()): out[i] = (float(lp[k, 1] - lp[k, 0]), float(lp[k, 1].exp()), int(logits[k].argmax()))
    return out


def summarize(rows, preds):
    y = [int(r["label"]) for r in rows]; s = [preds[i][0] for i in range(len(rows))]; am = [preds[i][2] for i in range(len(rows))]
    gen = [i for i, r in enumerate(rows) if r["label"] == "0"]
    res = dict(n=len(rows), n_genuine=len(gen), auc=auc(y, s), accuracy=float(np.mean([a == t for a, t in zip(am, y)])),
               genuine_fp=int(sum(am[i] == 1 for i in gen)), altered_misses=int(sum(am[i] == 0 for i, r in enumerate(rows) if r["label"] == "1")))
    thr = float(np.quantile([s[i] for i in gen], .9))
    for fam in sorted({r["family"] for r in rows} - {"genuine"}):
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        res[fam] = dict(n=len(idx), auc=auc([0] * len(gen) + [1] * len(idx), [s[i] for i in gen] + [s[i] for i in idx]),
                        tpr_fpr10=float(np.mean([s[i] > thr for i in idx])), misses=int(sum(am[i] == 0 for i in idx)))
    alt = [i for i, r in enumerate(rows) if r["label"] == "1"]; res["tpr_fpr10"] = float(np.mean([s[i] > thr for i in alt]))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--raw-root", required=True)
    ap.add_argument("--model", required=True, choices=["resnet50", "efficientnet-b3"]); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(); seed_all(a.seed); run = Path(a.run_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    records = {r["image_id"]: r for r in csv.DictReader(open(run / "inputs" / "records.csv"))}
    split = list(csv.DictReader(open(run / "inputs" / "split.csv")))
    enrol = json.load(open(run / "preparation" / "target-enrollment.json")); query_ids = set(enrol["query_ids"])

    def rows_for(ids):
        rs = []
        for i in ids:
            rec = records[i]; fam = rec["attack_family"] or "genuine"
            rs.append(dict(image_id=i, path=rec["path"], label=rec["label"], family=fam, design=rec["design_id"]))
        return rs

    train = rows_for([r["image_id"] for r in split if r["role"] == "train"]); val = rows_for([r["image_id"] for r in split if r["role"] == "val"])
    test = rows_for(sorted(i for i in (r["image_id"] for r in split if r["role"] == "test") if i in query_ids))
    assert len(test) == enrol["query_count"], (len(test), enrol["query_count"])
    print(f"group {enrol['group']} ({enrol['group_name']}): train {len(train)} val {len(val)} queries {len(test)} "
          f"(genuine {sum(r['label'] == '0' for r in test)}); model {a.model}", flush=True)
    dev = torch.device("cuda"); model = build_model(a.model).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.9, patience=1, eps=1e-6)
    crit = nn.CrossEntropyLoss()
    tr_ds = Imgs(train, a.raw_root, True, seed=a.seed); va_ds = Imgs(val, a.raw_root, False)
    tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=False,
                    generator=torch.Generator().manual_seed(a.seed), persistent_workers=False)
    va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
    best = (-1.0, None); log = open(out / "train_log.csv", "w", newline=""); w = csv.writer(log); w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auc", "lr", "seconds"])
    for ep in range(a.epochs):
        t0 = time.time(); tr_ds.epoch = ep; model.train(); tl = 0.0; n = 0
        for x, y, _ in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True); opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step(); tl += float(loss) * len(y); n += len(y)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y, _ in va: vl += float(crit(model(x.to(dev)), y.to(dev))) * len(y)
        vp = score(model, va, dev); vacc = float(np.mean([vp[i][2] == int(val[i]["label"]) for i in range(len(val))]))
        vauc = auc([int(r["label"]) for r in val], [vp[i][0] for i in range(len(val))]); sched.step(vl / len(val))
        w.writerow([ep, tl / n, vl / len(val), vacc, vauc, opt.param_groups[0]["lr"], round(time.time() - t0, 1)]); log.flush()
        print(f"epoch {ep} train {tl / n:.4f} val {vl / len(val):.4f} acc {vacc:.4f} auc {vauc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vacc > best[0]:
            best = (vacc, ep); torch.save(dict(state_dict=model.state_dict(), epoch=ep, val_acc=vacc, val_auc=vauc, model=a.model), out / "best.pt")
    log.close(); print("best val acc", best, flush=True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["state_dict"]); summary = dict(model=a.model, group=enrol["group"],
                                                                                                 group_name=enrol["group_name"], selected_epoch=best[1], val_acc=best[0])
    for variant, q in (("native", None), ("q75", 75)):
        loader = DataLoader(Imgs(test, a.raw_root, False, quality=q), batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
        preds = score(model, loader, dev); summary[variant] = summarize(test, preds)
        with (out / f"{variant}_predictions.csv").open("w", newline="") as h:
            cw = csv.DictWriter(h, fieldnames=["image_id", "label", "family", "design", "attack_logit", "p_altered"]); cw.writeheader()
            for i, r in enumerate(test): cw.writerow(dict(image_id=r["image_id"], label=r["label"], family=r["family"], design=r["design"], attack_logit=preds[i][0], p_altered=preds[i][1]))
        s = summary[variant]; print(variant, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)}),
                                    {k: round(v["auc"], 3) for k, v in s.items() if isinstance(v, dict)}, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
