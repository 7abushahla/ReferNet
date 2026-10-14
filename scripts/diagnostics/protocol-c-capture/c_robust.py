"""DIAGNOSTIC (Protocol C root cause, zero-shot): SIDTD-only training of the study OFF / ON networks with
*independent capture simulation* on the query (blur, sensor noise, colour cast and illumination ramp, JPEG, sub-pixel
perspective misregistration; a blurred reference now and then for the ON arm) and *photometric alignment* of the
warped reference to the query on the static overlap.  No FantasyID image is used for training or selection
(selection = SIDTD fold-0 validation, accuracy then AUC, as in the study).

Scoring: SIDTD fold-0 test (native, Q75) and the 467 locked Protocol C queries with
  * the locked enrollment products and similarity registrations (bank variant "locked"), and
  * rebuilt banks: homography registration, one bank per design x capture device, queries registered by homography
    to the bank of their own device (variant "fixed").  Banks are built from the locked C support manifests only
    (reference-only genuine captures of the three test designs; the test queries never touch bank construction).
Optionally (--score-only --checkpoint) rescore an existing study checkpoint under both variants.

Not a protocol result: no receipts, no locked authorization.  Same model, optimizer, schedule, coherent geometric
augmentation and selection rule as the study; tensors built by the repository's own functions where the geometry
allows (to_canvas, _features, _valid_median, _contributor, _photometric, _normalize).
"""
import argparse, csv, hashlib, json, math, random, time
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from refer_or_clear.data.rectify import to_canvas, CANVAS_H, CANVAS_W
from refer_or_clear.data.enrollment import _valid_median, _contributor, ZONE_DILATE_PX
from refer_or_clear.data.register import _features, RATIO, RANSAC_PX, MIN_INLIERS
from refer_or_clear.data.direct_reference import _available
from refer_or_clear.data.dataset import _photometric, _normalize
from refer_or_clear.models.direct_reference import build_direct_reference_model

SEED = 42


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int); p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n): return None
    return float((p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean())


def reencode(bgr, q):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)]); assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE); assert m is not None and m.shape == (CANVAS_H, CANVAS_W), path
    return m


# ------------------------------------------------------------------------------------------------- geometry (general)
def matches(fs, ft):
    k1, d1 = fs; k2, d2 = ft
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12: return None, None
    good = [a for a, b in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if a.distance < RATIO * b.distance]
    if len(good) < 12: return None, None
    return np.float32([k1[m.queryIdx].pt for m in good]), np.float32([k2[m.trainIdx].pt for m in good])


def fit(fs, ft, model, seed=0):
    cv2.setRNGSeed(seed)
    p1, p2 = matches(fs, ft)
    if p1 is None: return None, 0
    if model == "similarity":
        A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        M = None if A is None else np.vstack([A, [0, 0, 1]])
    else:
        M, inl = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
    n = 0 if inl is None else int(np.asarray(inl).sum())
    if M is None or n < MIN_INLIERS or not np.isfinite(M).all(): return None, n
    return np.asarray(M, np.float64), n


def warp(img, M, mask=False):
    return cv2.warpPerspective(img, np.asarray(M, np.float64), (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def polygon_mask(rectangles, M):
    mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    for _, left, top, w, h in rectangles:
        pts = np.float32([[left, top], [left + w, top], [left + w, top + h], [left, top + h]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, np.asarray(M, np.float64)).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(mapped * 256).astype(np.int32)], 255, shift=8)
    return mask


def overlap_of(qvalid, masks, M):
    inv = np.linalg.inv(np.asarray(M, np.float64))
    ov = (qvalid > 0) & (warp(masks["target_valid"], inv, True) > 0) & (warp(masks["valid"], inv, True) > 0)
    return inv, ov.astype(np.uint8)


def build_bank(items, model, out_dir, seed=0):
    """items: list of dicts(canvas, valid, native, feats, rectangles). Two-pass median exactly as enrollment._build_bank, general warp."""
    anchor = items[0]; first_i, first_v = [], []
    for it in items:
        M = np.eye(3) if it is anchor else fit(it["feats"], anchor["feats"], model, seed)[0]
        if M is not None: first_i.append(warp(it["canvas"], M)); first_v.append(warp(it["valid"], M, True))
    target, tcov = _valid_median(first_i, first_v); target_valid = (tcov > 0).astype(np.uint8) * 255
    tf = _features(target); final_i, final_v, used = [], [], 0; zone = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    for it in items:
        M, _ = fit(it["feats"], tf, model, seed + 1)
        if M is None: continue
        v = cv2.bitwise_and(warp(it["valid"], M, True), target_valid)
        if not v.any(): continue
        final_i.append(warp(it["canvas"], M)); final_v.append(v); used += 1
        zone |= cv2.bitwise_and(polygon_mask(it["rectangles"], M @ it["native"]), v)
    assert used, "empty bank"
    median, cov = _valid_median(final_i, final_v); valid = (cov > 0).astype(np.uint8) * 255
    zone = cv2.bitwise_and(cv2.dilate(zone, np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8)), valid)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, im in (("target", target), ("target_valid", target_valid), ("median", median), ("valid", valid), ("zone", zone)):
        assert cv2.imwrite(str(out_dir / f"{name}.png"), im)
    (out_dir / "geometry.json").write_text(json.dumps(dict(model=model, n=len(items), registered=used)))
    return {n: str(out_dir / f"{n}.png") for n in ("target", "target_valid", "median", "valid", "zone")}, used


def device_of(path):
    for d in ("huawei", "iphone15pro", "iphone15", "scan"):
        if f"/{d}/" in path: return d
    return "digital"


# ----------------------------------------------------------------------------------------------- capture simulation
def capture_sim(q, rng):
    """Independent physical-capture simulation on the query canvas (uint8 BGR). Each component with its own probability."""
    x = q.astype(np.float32)
    if rng.random() < .5:  # colour cast + exposure + illumination ramp
        gain = np.float32([rng.uniform(.85, 1.15) for _ in range(3)]); off = rng.uniform(-12, 12)
        x = x * gain + off
        g = rng.uniform(-.25, .25); ramp = np.linspace(-.5, .5, CANVAS_W, dtype=np.float32)[None, :] if rng.random() < .5 else np.linspace(-.5, .5, CANVAS_H, dtype=np.float32)[:, None]
        x = x * (1 + g * ramp)[..., None]
    if rng.random() < .5:  # optical blur
        s = rng.uniform(.3, 1.5); x = cv2.GaussianBlur(x, (0, 0), s)
    if rng.random() < .5:  # sensor / print noise
        s = rng.uniform(2, 8); x = x + np.random.default_rng(rng.getrandbits(32)).normal(0, s, x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < .5:  # sub-pixel to few-pixel misregistration the similarity model would leave behind
        src = np.float32([[0, 0], [CANVAS_W, 0], [CANVAS_W, CANVAS_H], [0, CANVAS_H]])
        dst = src + np.float32([[rng.uniform(-3, 3), rng.uniform(-3, 3)] for _ in range(4)])
        x = cv2.warpPerspective(x, cv2.getPerspectiveTransform(src, dst), (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .5:  # re-encoding
        x = reencode(x, rng.randint(45, 90))
    return x


def photo_align(reference, query, overlap, zone):
    """Per-channel linear map of the warped reference onto the query, fitted on the static overlap (overlap minus zone)."""
    static = (overlap > 0) & (zone == 0)
    if static.sum() < 500: return reference
    idx = np.flatnonzero(static.ravel()); idx = idx[:: max(1, len(idx) // 20000)]
    r = reference.reshape(-1, 3)[idx].astype(np.float32); q = query.reshape(-1, 3)[idx].astype(np.float32); out = reference.astype(np.float32).copy()
    for c in range(3):
        a, b = np.polyfit(r[:, c], q[:, c], 1)
        a = float(np.clip(a, .5, 2.)); out[..., c] = a * out[..., c] + b
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------------------- dataset
class Entries(Dataset):
    """entries: dicts with path, label, family, dataset, capture, products, matrix (3x3 or None), image_id."""

    def __init__(self, entries, raw_root, reference_enabled, train, quality=None, seed=SEED, capture_aug=True, align=True):
        self.entries, self.raw_root, self.reference_enabled, self.train, self.quality = entries, Path(raw_root), reference_enabled, train, quality
        self.capture_aug, self.align, self.rng, self._masks = capture_aug, align, random.Random(seed), {}

    def __len__(self): return len(self.entries)

    def masks(self, products):
        key = products["zone"]
        if key not in self._masks: self._masks[key] = {n: read_mask(products[n]) for n in ("target_valid", "valid", "zone")}
        return self._masks[key]

    def __getitem__(self, i):
        e = self.entries[i]
        raw = cv2.imread(str(self.raw_root / e["path"]), cv2.IMREAD_COLOR); assert raw is not None, e["path"]
        if self.quality: raw = reencode(raw, self.quality)
        query, qvalid, _ = to_canvas(raw)
        overlap, zone, reference = np.zeros((CANVAS_H, CANVAS_W), np.uint8), np.zeros((CANVAS_H, CANVAS_W), np.uint8), None
        if e["matrix"] is not None:
            inverse, overlap = overlap_of(qvalid, self.masks(e["products"]), np.asarray(e["matrix"]))
            zone = (warp(self.masks(e["products"])["zone"], inverse, True) > 0).astype(np.uint8) * overlap
            if self.reference_enabled and _available(overlap):
                median = cv2.imread(e["products"]["median"], cv2.IMREAD_COLOR); assert median.shape == (CANVAS_H, CANVAS_W, 3)
                reference = warp(median, inverse)
        if self.train:  # study augmentation: coherent small affine on everything, 25% 2-px query jitter, photometric on query
            rng = self.rng
            ang = rng.uniform(-1., 1.); tx, ty = rng.uniform(-8, 8), rng.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((CANVAS_W / 2, CANVAS_H / 2), ang, 1.); M[:, 2] += (tx, ty)
            wi = lambda a: cv2.warpAffine(a, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            wm = lambda a: cv2.warpAffine(a, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            query, zone, overlap = wi(query), wm(zone), wm(overlap)
            if reference is not None: reference = wi(reference)
            if rng.random() < .25:
                J = np.float32([[1, 0, rng.randint(-2, 2)], [0, 1, rng.randint(-2, 2)]])
                query = cv2.warpAffine(query, J, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if reference is not None: reference = cv2.warpAffine(reference, J, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                zone = cv2.warpAffine(zone, J, (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                overlap = cv2.warpAffine(overlap, J, (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            st = random.getstate(); random.seed(rng.random()); query = _photometric(query, .2); random.setstate(st)
            if self.capture_aug:
                query = capture_sim(query, rng)
                if reference is not None and rng.random() < .3: reference = cv2.GaussianBlur(reference, (0, 0), rng.uniform(.5, 2.))
        available = _available(overlap)
        if not available: zone.fill(0); overlap.fill(0)
        zone *= overlap
        if reference is not None and available and self.align: reference = photo_align(reference, query, overlap, zone)
        out = dict(x=torch.from_numpy(np.ascontiguousarray(np.concatenate([_normalize(query), zone[None].astype(np.float32)]))),
                   overlap=torch.from_numpy(overlap[None].astype(np.float32)), available=torch.tensor(available, dtype=torch.bool),
                   y=torch.tensor(float(e["label"])), image_id=e["image_id"], family=e["family"], dataset=e["dataset"], capture=e["capture"])
        if self.reference_enabled:
            out["reference"] = torch.from_numpy(np.ascontiguousarray(_normalize(reference))) if (available and reference is not None) else torch.zeros((3, CANVAS_H, CANVAS_W), dtype=torch.float32)
        return out


def forward(model, b, dev, reference_enabled):
    return model(b["x"].to(dev), overlap=b["overlap"].to(dev), reference=b["reference"].to(dev) if reference_enabled else None, context_available=b["available"].to(dev))


def run(model, loader, dev, reference_enabled):
    model.eval(); out = []
    with torch.inference_mode():
        for b in loader:
            o = forward(model, b, dev, reference_enabled)
            for k in range(b["x"].shape[0]):
                out.append(dict(image_id=b["image_id"][k], label=int(b["y"][k]), family=b["family"][k], dataset=b["dataset"][k], capture=b["capture"][k],
                                available=bool(b["available"][k]), attack_logit=float(o["attack_logit"][k]), query_logit=float(o["query_logit"][k]),
                                correction_logit=float(o["correction_logit"][k])))
    return out


def summarize(rows):
    y = [r["label"] for r in rows]; l = [r["attack_logit"] for r in rows]; gen = [r for r in rows if r["label"] == 0]
    res = dict(n=len(rows), n_genuine=len(gen), accuracy=float(np.mean([(v > 0) == bool(t) for v, t in zip(l, y)])), auc=auc(l, y),
               auc_query=auc([r["query_logit"] for r in rows], y), auc_correction=auc([r["correction_logit"] for r in rows], y),
               genuine_false_positives=int(sum(r["attack_logit"] > 0 for r in gen)), altered_misses=int(sum(r["attack_logit"] <= 0 for r in rows if r["label"] == 1)),
               unavailable=int(sum(not r["available"] for r in rows)))
    for key in ("family", "capture"):
        for f in sorted({r[key] for r in rows if r[key]}):
            sub = [r for r in rows if r[key] == f]
            if key == "family": sub = gen + [r for r in sub if r["label"] == 1]
            if len({r["label"] for r in sub}) < 2: continue
            res[f"{key}:{f}"] = dict(n=len(sub), auc=auc([r["attack_logit"] for r in sub], [r["label"] for r in sub]),
                                      accuracy=float(np.mean([(r["attack_logit"] > 0) == bool(r["label"]) for r in sub])))
    return res


def load_state(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck.get("model_state_dict", ck))) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys {unexpected[:5]}"
    assert not missing, f"missing keys {missing[:5]}"
    print("loaded", path, "epoch", ck.get("epoch") if isinstance(ck, dict) else None, "missing", len(missing), flush=True)
    return ck.get("epoch") if isinstance(ck, dict) else None


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--raw-root", required=True)
    ap.add_argument("--a-registration", required=True); ap.add_argument("--a-eval-registration", required=True)
    ap.add_argument("--c-registration", required=True); ap.add_argument("--c-labels", required=True); ap.add_argument("--c-support", required=True, help="groupC enrollment root")
    ap.add_argument("--weights", required=True); ap.add_argument("--weights-sha256", required=True); ap.add_argument("--out", required=True); ap.add_argument("--banks", required=True)
    ap.add_argument("--arm", choices=("off", "on"), required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-capture-aug", action="store_true"); ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--score-only", action="store_true"); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); banks_dir = Path(a.banks); banks_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    raw_root = Path(a.raw_root); ref_on = a.arm == "on"; align = not a.no_align; cap_aug = not a.no_capture_aug
    rec = {r["image_id"]: r for r in csv.DictReader(open(a.records))}

    def sidtd_entries(reg_path, roles):
        reg = json.load(open(reg_path))["entries"]; res = []
        for k, v in reg.items():
            if v["role"] not in roles: continue
            r = rec[k]; prod = {n: v["products"][n]["path"] for n in ("target", "target_valid", "valid", "zone", "median")}
            res.append(dict(image_id=k, path=v["path"], label=int(r["label"]), family=r["attack_family"] if r["label"] == "1" else "genuine",
                            dataset="sidtd", capture="template", products=prod, matrix=v["query_canvas_to_bank"] if v["status"] == "fit" else None, role=v["role"]))
        return res
    a_train = sidtd_entries(a.a_registration, {"train"}); a_val = sidtd_entries(a.a_registration, {"val"}); a_test = sidtd_entries(a.a_eval_registration, {"test"})
    print("sidtd", len(a_train), len(a_val), len(a_test), flush=True)

    # ---- Protocol C queries: locked variant (products + similarity matrices from the locked registration)
    labels = {r["image_id"]: r for r in csv.DictReader(open(a.c_labels))}
    creg = json.load(open(a.c_registration))["entries"]; c_locked = []
    for k, v in creg.items():
        prod = {n: v["products"][n]["path"] for n in ("target", "target_valid", "valid", "zone", "median")}
        c_locked.append(dict(image_id=k, path=v["path"], label=int(labels[k]["label"]), family=labels[k]["attack_family"] or "genuine", dataset="fantasyid-test",
                             capture=device_of(v["path"]), products=prod, matrix=v["query_canvas_to_bank"] if v["status"] == "fit" else None, role="test", design=v["design_id"]))
    # ---- fixed variant: homography, one bank per design x device from the locked support manifests; queries registered by homography
    reg_cache_path = banks_dir / "registrations.json"; reg_cache = json.load(open(reg_cache_path)) if reg_cache_path.exists() else {}
    c_fixed = []
    for design in sorted({e["design"] for e in c_locked}):
        man = json.load(open(Path(a.c_support) / design / "enrollment" / "manifest.json"))
        items = []
        for iid in man["support_ids"]:
            meta = man["contributors"][iid]; _, rects, _ = _contributor(raw_root, meta)
            img = cv2.imread(str(raw_root / meta["path"]), cv2.IMREAD_COLOR); assert img is not None
            canvas, valid, native = to_canvas(img)
            items.append(dict(canvas=canvas, valid=valid, native=native, feats=_features(canvas), rectangles=rects, device=device_of(meta["path"])))
        banks = {}
        for dev_name in sorted({it["device"] for it in items}) + ["mixed"]:
            sub = items if dev_name == "mixed" else [it for it in items if it["device"] == dev_name]
            bd = banks_dir / design.replace("/", "_") / dev_name
            if (bd / "zone.png").exists(): banks[dev_name] = {n: str(bd / f"{n}.png") for n in ("target", "target_valid", "median", "valid", "zone")}; used = "cached"
            else: banks[dev_name], used = build_bank(sub, "homography", bd, SEED)
            print(f"bank {design}/{dev_name}: {len(sub)} contributors, registered {used}", flush=True)
        tfeats = {k: _features(cv2.imread(b["target"], cv2.IMREAD_COLOR)) for k, b in banks.items()}
        bmasks = {k: {n: read_mask(b[n]) for n in ("target_valid", "valid", "zone")} for k, b in banks.items()}
        for e in [e for e in c_locked if e["design"] == design]:
            bank_name = e["capture"] if e["capture"] in banks else "mixed"
            key = f"{e['image_id']}|{bank_name}"
            if key not in reg_cache:
                raw = cv2.imread(str(raw_root / e["path"]), cv2.IMREAD_COLOR); canvas, qvalid, _ = to_canvas(raw)
                M, inl = fit(_features(canvas), tfeats[bank_name], "homography", SEED)
                ok = M is not None and _available(overlap_of(qvalid, bmasks[bank_name], M)[1])
                reg_cache[key] = dict(matrix=M.tolist() if ok else None, inliers=inl)
            c_fixed.append(dict(e, products=banks[bank_name], matrix=reg_cache[key]["matrix"], bank=bank_name))
        reg_cache_path.write_text(json.dumps(reg_cache))
    print("C queries", len(c_locked), "fixed unregistered", sum(e["matrix"] is None for e in c_fixed), flush=True)

    train_entries, val_entries = a_train, a_val
    if a.smoke:
        train_entries = train_entries[:a.batch * 4] + train_entries[-a.batch * 4:]; val_entries = val_entries[:16] + val_entries[-16:]; a_test = a_test[:16]
        c_locked = c_locked[:16] + c_locked[-16:]; c_fixed = c_fixed[:16] + c_fixed[-16:]; a.epochs = 1
    dev = torch.device("cuda")
    model = build_direct_reference_model(Path(a.weights), a.weights_sha256, reference_enabled=ref_on, head_seed=SEED, adapter_seed=SEED,
                                         local_enabled=ref_on, reference_bn="shared").to(dev)
    mk = lambda ents, train, q=None, seed=SEED: DataLoader(Entries(ents, raw_root, ref_on, train, quality=q, seed=seed, capture_aug=cap_aug, align=align), batch_size=a.batch,
                                                           shuffle=train, num_workers=a.workers, drop_last=train, persistent_workers=a.workers > 0 and train)
    result = dict(arm=a.arm, epochs=a.epochs, batch=a.batch, capture_aug=cap_aug, align=align, score_only=a.score_only, checkpoint=a.checkpoint,
                  n_train=len(train_entries), n_val=len(val_entries), weights_sha256=a.weights_sha256,
                  c_registration_sha256=hashlib.sha256(Path(a.c_registration).read_bytes()).hexdigest())
    if a.score_only:
        result["selected_epoch"] = load_state(model, a.checkpoint)
    else:
        pretrained, new = model.parameter_groups()
        opt = torch.optim.AdamW([dict(params=pretrained, lr=1e-4), dict(params=new, lr=1e-3)], weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
        train_loader = mk(train_entries, True); val_loader = mk(val_entries, False)
        total = a.epochs * len(train_loader); warm = min(1000, int(0.1 * total)); base = [g["lr"] for g in opt.param_groups]
        crit = nn.BCEWithLogitsLoss(); step = 0; best = None; log = []
        print(dict(arm=a.arm, train=len(train_entries), val=len(val_entries), iters_per_epoch=len(train_loader)), flush=True)
        for ep in range(1, a.epochs + 1):
            model.train(); t0 = time.time(); losses = []
            for b in train_loader:
                f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g, lr in zip(opt.param_groups, base): g["lr"] = lr * f
                o = forward(model, b, dev, ref_on); loss = crit(o["attack_logit"], b["y"].to(dev))
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1; losses.append(loss.item())
            v = summarize(run(model, val_loader, dev, ref_on))
            row = dict(epoch=ep, train_loss=float(np.mean(losses)), val_acc=v["accuracy"], val_auc=v["auc"], seconds=time.time() - t0); log.append(row); print(json.dumps(row), flush=True)
            key = (v["accuracy"], v["auc"], -ep)
            if best is None or key > best[0]:
                best = (key, ep); torch.save(dict(epoch=ep, state_dict=model.state_dict(), val=v), out / "best.pt")
        with (out / "log.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(ck["state_dict"]); result["selected_epoch"] = ck["epoch"]
    for name, ents, q in (("val", val_entries, None), ("a_test", a_test, None), ("a_test_q75", a_test, 75),
                          ("c_locked", c_locked, None), ("c_fixed", c_fixed, None), ("c_fixed_q75", c_fixed, 75)):
        rows = run(model, mk(ents, False, q), dev, ref_on); result[name] = summarize(rows)
        with (out / f"{name}_predictions.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in result[name].items() if not isinstance(v, dict)}), flush=True)
        print("   ", json.dumps({k: (round(v["auc"], 3) if v["auc"] is not None else None) for k, v in result[name].items() if isinstance(v, dict)}), flush=True)
    (out / "summary.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
