"""DIAGNOSTIC (compression-cue sensitivity): train OUR OFF network (the study's DirectReferenceClassifier with the
reference path disabled: LEVIR-initialised ChangerEx ResNet-18 encoder + global-pool classifier) on the RAW SIDTD files
resized to a square, i.e. the published input path that keeps the JPEG quantization cue, instead of the registered
design canvas. Then score the selected checkpoint on the fold-0 validation and test partitions, native and re-encoded
at IJG quality 75. Not a protocol result: no receipts, no locked authorization; it is the sensitivity row of Protocol A.
"""
import argparse, csv, hashlib, json, math, random, time
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from refer_or_clear.models.direct_reference import build_direct_reference_model

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int); p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n): return None
    return float((p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean())


def reencode(bgr, q):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q]); assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def to_tensor(bgr, size):
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)
    return np.concatenate([x, np.ones((1, size, size), np.float32)], 0)  # zone channel = whole image


class RawSIDTD(Dataset):
    def __init__(self, rows, raw_root, size, train, quality=None, seed=42):
        self.rows, self.raw_root, self.size, self.train, self.quality = rows, Path(raw_root), size, train, quality
        self.rng = random.Random(seed)

    def __len__(self): return len(self.rows)

    def _augment(self, bgr):  # the published raw-RGB preset: random resized crop (scale .8-1, ratio 3/4-4/3), flips
        h, w = bgr.shape[:2]; area = h * w
        for _ in range(10):
            s = self.rng.uniform(.8, 1.) * area; r = math.exp(self.rng.uniform(math.log(3 / 4), math.log(4 / 3)))
            cw, ch = int(round(math.sqrt(s * r))), int(round(math.sqrt(s / r)))
            if 0 < cw <= w and 0 < ch <= h:
                x0, y0 = self.rng.randint(0, w - cw), self.rng.randint(0, h - ch); bgr = bgr[y0:y0 + ch, x0:x0 + cw]; break
        if self.rng.random() < .5: bgr = bgr[:, ::-1]
        if self.rng.random() < .5: bgr = bgr[::-1, :]
        return np.ascontiguousarray(bgr)

    def __getitem__(self, i):
        r = self.rows[i]
        bgr = cv2.imread(str(self.raw_root / r["path"]), cv2.IMREAD_COLOR); assert bgr is not None, r["path"]
        if self.quality: bgr = reencode(bgr, self.quality)  # before any resizing, as in the paper's control
        if self.train: bgr = self._augment(bgr)
        return dict(x=torch.from_numpy(to_tensor(bgr, self.size)), y=torch.tensor(float(r["label"])),
                    image_id=r["image_id"], family=r["attack_family"] if int(r["label"]) else "genuine")


def run(model, loader, dev):
    model.eval(); out = []
    with torch.inference_mode():
        for b in loader:
            x = b["x"].to(dev); n = x.shape[0]
            o = model(x, overlap=torch.ones(n, 1, x.shape[2], x.shape[3], device=dev), reference=None,
                      context_available=torch.zeros(n, dtype=torch.bool, device=dev))
            for k in range(n):
                out.append(dict(image_id=b["image_id"][k], label=int(b["y"][k]), family=b["family"][k], attack_logit=float(o["attack_logit"][k])))
    return out


def summarize(rows):
    y = [r["label"] for r in rows]; l = [r["attack_logit"] for r in rows]
    gen = [r for r in rows if r["label"] == 0]
    res = dict(n=len(rows), n_genuine=len(gen), accuracy=float(np.mean([(v > 0) == bool(t) for v, t in zip(l, y)])), auc=auc(l, y),
               genuine_false_positives=int(sum(r["attack_logit"] > 0 for r in gen)),
               altered_misses=int(sum(r["attack_logit"] <= 0 for r in rows if r["label"] == 1)),
               mean_logit_genuine=float(np.mean([r["attack_logit"] for r in gen])),
               mean_logit_altered=float(np.mean([r["attack_logit"] for r in rows if r["label"] == 1])))
    for f in sorted({r["family"] for r in rows if r["label"] == 1}):
        sub = [r for r in rows if r["family"] == f]
        res[f] = dict(n=len(sub), recall=float(np.mean([r["attack_logit"] > 0 for r in sub])),
                      auc=auc([r["attack_logit"] for r in gen + sub], [r["label"] for r in gen + sub]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--split", required=True); ap.add_argument("--raw-root", required=True)
    ap.add_argument("--weights", required=True); ap.add_argument("--weights-sha256", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--size", type=int, default=320); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rec = {r["image_id"]: r for r in csv.DictReader(open(a.records))}
    split = list(csv.DictReader(open(a.split)))
    roles = {role: [dict(rec[r["image_id"]], label=r["label"]) for r in split if r["role"] == role] for role in ("train", "val", "test")}
    if a.smoke: roles = {k: v[:a.batch * 2] for k, v in roles.items()}; a.epochs = 2
    print({k: len(v) for k, v in roles.items()}, flush=True)
    dev = torch.device("cuda")
    model = build_direct_reference_model(Path(a.weights), a.weights_sha256, reference_enabled=False, head_seed=a.seed,
                                         adapter_seed=a.seed, local_enabled=False, reference_bn="shared").to(dev)
    pretrained, new = model.parameter_groups()
    opt = torch.optim.AdamW([dict(params=pretrained, lr=1e-4), dict(params=new, lr=1e-3)], weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
    train_loader = DataLoader(RawSIDTD(roles["train"], a.raw_root, a.size, True, seed=a.seed), batch_size=a.batch, shuffle=True,
                              num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    val_loader = DataLoader(RawSIDTD(roles["val"], a.raw_root, a.size, False), batch_size=a.batch, shuffle=False, num_workers=a.workers)
    total = a.epochs * len(train_loader); warm = min(1000, int(0.1 * total)); base = [g["lr"] for g in opt.param_groups]
    crit = nn.BCEWithLogitsLoss(); step = 0; best = None; log = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for b in train_loader:
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, lr in zip(opt.param_groups, base): g["lr"] = lr * f
            x, y = b["x"].to(dev), b["y"].to(dev); n = x.shape[0]
            o = model(x, overlap=torch.ones(n, 1, x.shape[2], x.shape[3], device=dev), reference=None, context_available=torch.zeros(n, dtype=torch.bool, device=dev))
            loss = crit(o["attack_logit"], y); opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1; losses.append(loss.item())
        v = summarize(run(model, val_loader, dev))
        row = dict(epoch=ep, train_loss=float(np.mean(losses)), val_acc=v["accuracy"], val_auc=v["auc"], seconds=time.time() - t0); log.append(row)
        print(json.dumps(row), flush=True)
        key = (v["accuracy"], v["auc"], -ep)
        if best is None or key > best[0]:
            best = (key, ep); torch.save(dict(epoch=ep, state_dict=model.state_dict(), val=v), out / "best.pt")
    with (out / "log.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(ck["state_dict"])
    result = dict(selected_epoch=ck["epoch"], size=a.size, epochs=a.epochs, batch=a.batch, seed=a.seed,
                  input="raw file resized to %dx%d (INTER_LINEAR), no registration; zone channel = 1" % (a.size, a.size),
                  model="study OFF arm: DirectReferenceClassifier(reference_enabled=False, local_enabled=False), LEVIR init",
                  weights_sha256=a.weights_sha256, records_sha256=hashlib.sha256(Path(a.records).read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(Path(a.split).read_bytes()).hexdigest())
    for role in ("val", "test"):
        for name, q in (("native", None), ("q75", 75)):
            rows = run(model, DataLoader(RawSIDTD(roles[role], a.raw_root, a.size, False, quality=q), batch_size=a.batch, shuffle=False, num_workers=a.workers), dev)
            result[f"{role}_{name}"] = summarize(rows)
            with (out / f"{role}_{name}_predictions.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            print(role, name, json.dumps({k: v for k, v in result[f"{role}_{name}"].items() if not isinstance(v, dict)}), flush=True)
    (out / "summary.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
