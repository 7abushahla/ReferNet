"""Registration-model probe (CPU only, no model inference).

Question: how much of the query-vs-reference residual on physically captured cards comes from the
geometric model (4-DOF similarity) and from mixing capture devices in one median bank?

Non-test diagnostic: the enrollment support of each design (reference-only genuine captures) is split
into two identity halves; a bank is built from one half and the other half's captures are registered to
it. Residual outside the variable-field zone is measured per device, for each registration model and
bank device policy. The 467 Protocol C test queries are scored DESCRIPTIVELY against the full bank only.
SIDTD alb_id fold 0 is the control (digital renders).
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, cv2

cv2.setNumThreads(2)
from refer_or_clear.data.rectify import to_canvas, CANVAS_H, CANVAS_W
from refer_or_clear.data.register import _features, RATIO, RANSAC_PX, MIN_INLIERS
from refer_or_clear.data.enrollment import _contributor, _valid_median, ZONE_DILATE_PX

MODELS = ("similarity", "homography")


def matches(fs, ft):
    k1, d1 = fs; k2, d2 = ft
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < RATIO * b.distance]
    if len(good) < 12:
        return None, None
    return np.float32([k1[m.queryIdx].pt for m in good]), np.float32([k2[m.trainIdx].pt for m in good])


def fit(fs, ft, model, seed=0):
    cv2.setRNGSeed(seed)
    p1, p2 = matches(fs, ft)
    if p1 is None:
        return None, 0
    if model == "similarity":
        A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        M = None if A is None else np.vstack([A, [0, 0, 1]])
    elif model == "affine":
        A, inl = cv2.estimateAffine2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        M = None if A is None else np.vstack([A, [0, 0, 1]])
    else:
        M, inl = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
    n = 0 if inl is None else int(np.asarray(inl).sum())
    if M is None or n < MIN_INLIERS or not np.isfinite(M).all():
        return None, n
    return np.asarray(M, np.float64), n


def warp(img, M, mask=False):
    return cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def polygon_mask(rectangles, M):
    mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    for _, left, top, w, h in rectangles:
        pts = np.float32([[left, top], [left + w, top], [left + w, top + h], [left, top + h]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        cv2.fillPoly(mask, [np.round(mapped * 256).astype(np.int32)], 255, shift=8)
    return mask


class Item:
    def __init__(self, image_id, path, rectangles, device, label, family, raw_root):
        self.image_id, self.path, self.rectangles, self.device, self.label, self.family = image_id, path, rectangles, device, label, family
        img = cv2.imread(str(raw_root / path), cv2.IMREAD_COLOR)
        assert img is not None, path
        self.canvas, self.valid, self.native = to_canvas(img)
        self.feats = _features(self.canvas)


def build_bank(items, model, seed=0):
    anchor = items[0]
    first_i, first_v = [], []
    for it in items:
        M = np.eye(3) if it is anchor else fit(it.feats, anchor.feats, model, seed)[0]
        if M is not None:
            first_i.append(warp(it.canvas, M)); first_v.append(warp(it.valid, M, True))
    target, tcov = _valid_median(first_i, first_v)
    target_valid = (tcov > 0).astype(np.uint8) * 255
    tf = _features(target)
    final_i, final_v, used, zone = [], [], 0, np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    for it in items:
        M, _ = fit(it.feats, tf, model, seed + 1)
        if M is None:
            continue
        v = cv2.bitwise_and(warp(it.valid, M, True), target_valid)
        if not v.any():
            continue
        final_i.append(warp(it.canvas, M)); final_v.append(v); used += 1
        zone |= cv2.bitwise_and(polygon_mask(it.rectangles, M @ it.native), v)
    median, cov = _valid_median(final_i, final_v)
    valid = (cov > 0).astype(np.uint8) * 255
    zone = cv2.bitwise_and(cv2.dilate(zone, np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8)), valid)
    return dict(target=target, target_valid=target_valid, tf=tf, median=median, valid=valid, zone=zone, used=used, n=len(items))


def measure(it, bank, model, seed=0):
    M, inl = fit(it.feats, bank["tf"], model, seed)
    if M is None:
        return dict(inliers=inl, fit=0)
    inv = np.linalg.inv(M)
    overlap = (warp(bank["valid"], inv, True) > 0) & (it.valid > 0)
    zone = (warp(bank["zone"], inv, True) > 0) & overlap
    out = overlap & ~zone
    ref = warp(bank["median"], inv)
    q = it.canvas.astype(np.float32); r = ref.astype(np.float32)
    diff = np.abs(q - r).mean(axis=2)
    # photometric alignment: per-channel linear map of the reference onto the query, fitted on the static overlap
    r2 = r.copy()
    for c in range(3):
        a, b = np.polyfit(r[..., c][out], q[..., c][out], 1)
        r2[..., c] = a * r[..., c] + b
    diff2 = np.abs(q - r2).mean(axis=2)
    gq = cv2.cvtColor(it.canvas, cv2.COLOR_BGR2GRAY).astype(np.float32); gr = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(cv2.Laplacian(gq, cv2.CV_32F, ksize=3) - cv2.Laplacian(gr, cv2.CV_32F, ksize=3))
    return dict(inliers=inl, fit=1, overlap=float(overlap.mean()), zone=float(zone.mean()),
                out=float(diff[out].mean()), inz=float(diff[zone].mean()) if zone.any() else float("nan"),
                out_photo=float(diff2[out].mean()), hp_out=float(hp[out].mean()), hp_in=float(hp[zone].mean()) if zone.any() else float("nan"),
                ref=ref, overlap_mask=overlap, zone_mask=zone)


def device_of(path):
    for d in ("huawei", "iphone15pro", "iphone15", "scan"):
        if f"/{d}/" in path:
            return d
    return "digital"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True); ap.add_argument("--c-support", required=True, help="groupC enrollment root")
    ap.add_argument("--c-registration", required=True); ap.add_argument("--c-labels", required=True)
    ap.add_argument("--sidtd-manifest", required=True); ap.add_argument("--a-eval-registration", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--designs", default="chinese2,netherland,singapore")
    ap.add_argument("--max-queries", type=int, default=0)
    a = ap.parse_args()
    raw_root, out = Path(a.raw_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    def support_items(manifest):
        items = []
        for iid in manifest["support_ids"]:
            meta = manifest["contributors"][iid]
            _, rects, _ = _contributor(raw_root, meta)
            items.append(Item(iid, meta["path"], rects, device_of(meta["path"]), 0, "support", raw_root))
        return items

    def run_design(design, support, queries):
        print(f"[{time.time()-t0:6.0f}s] {design}: {len(support)} support, {len(queries)} queries", flush=True)
        key = lambda it: it.image_id if design.startswith("sidtd") else it.path.split("/")[-1].rsplit(".", 1)[0]
        identities = sorted({key(it) for it in support})
        half = {ident: i % 2 for i, ident in enumerate(identities)}
        for model in MODELS:
            # --- non-test: identity halves of the support, bank from one half, residual of the other half's captures
            for h in (0, 1):
                bank_items = [it for it in support if half[key(it)] == h]; probe_items = [it for it in support if half[key(it)] != h]
                banks = {"mixed": build_bank(bank_items, model)}
                for dev in sorted({it.device for it in bank_items}):
                    sub = [it for it in bank_items if it.device == dev]
                    if dev != "digital" and len(sub) >= 5:
                        banks[dev] = build_bank(sub, model)
                for it in probe_items:
                    for policy, bank in (("mixed", banks["mixed"]), ("same-device", banks.get(it.device))):
                        if bank is None:
                            continue
                        m = measure(it, bank, model)
                        rows.append(dict(design=design, split="support-half", model=model, policy=policy, bank_half=h, bank_n=bank["n"], bank_used=bank["used"],
                                         image_id=it.image_id, device=it.device, label=it.label, family=it.family,
                                         **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}))
            # --- descriptive: full bank, test queries
            full = build_bank(support, model)
            dev_banks = {}
            for dev in sorted({it.device for it in support}):
                sub = [it for it in support if it.device == dev]
                if dev != "digital" and len(sub) >= 5:
                    dev_banks[dev] = build_bank(sub, model)
            for it in queries:
                for policy, bank in (("mixed", full), ("same-device", dev_banks.get(it.device))):
                    if bank is None:
                        continue
                    m = measure(it, bank, model)
                    rows.append(dict(design=design, split="test-query", model=model, policy=policy, bank_half=-1, bank_n=bank["n"], bank_used=bank["used"],
                                     image_id=it.image_id, device=it.device, label=it.label, family=it.family,
                                     **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}))
                    if policy == "mixed" and m.get("fit") and it.family in ("genuine", "support") and it.device in ("scan", "huawei") and \
                            not (out / f"vis_{design.split('/')[-1]}_{it.device}_{model}.png").exists():
                        d = cv2.cvtColor(np.clip(np.abs(it.canvas.astype(np.float32) - m["ref"].astype(np.float32)).mean(axis=2) * 3, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(str(out / f"vis_{design.split('/')[-1]}_{it.device}_{model}.png"), np.concatenate([it.canvas, m["ref"], d], axis=1))
        print(f"[{time.time()-t0:6.0f}s] {design} done ({len(rows)} rows)", flush=True)

    # FantasyID test designs
    labels = {r["image_id"]: r for r in csv.DictReader(open(a.c_labels))}
    creg = json.load(open(a.c_registration))["entries"]
    for design in a.designs.split(","):
        man = json.load(open(Path(a.c_support) / "fantasyid" / design / "enrollment" / "manifest.json"))
        support = support_items(man)
        qs = []
        for iid, v in creg.items():
            if v["design_id"] != f"fantasyid/{design}":
                continue
            lab = labels[iid]
            qs.append((iid, v["path"], int(lab["label"]), lab["attack_family"] or "genuine"))
        if a.max_queries:
            qs = qs[: a.max_queries]
        queries = [Item(iid, path, [], device_of(path), lab, fam, raw_root) for iid, path, lab, fam in qs]
        run_design(f"fantasyid/{design}", support, queries)

    # SIDTD control
    man = json.load(open(a.sidtd_manifest))
    support = support_items(man)
    areg = json.load(open(a.a_eval_registration))["entries"]
    queries = []
    for iid, v in areg.items():
        if v.get("design_id") != man["design_id"]:
            continue
        queries.append(Item(iid, v["path"], [], "digital", int(v.get("label", -1)), v.get("attack_family") or "genuine", raw_root))
        if len(queries) >= 60:
            break
    run_design(man["design_id"], support, queries)

    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    # summary
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r.get("fit"):
            agg[(r["design"], r["split"], r["model"], r["policy"], r["device"], r["family"])].append(r)
    lines = ["design|split|model|policy|device|family|n|fit-fail|inliers|out|out_photo|in|hp_out"]
    for k in sorted(agg):
        rs = agg[k]; nf = sum(1 for r in rows if not r.get("fit") and (r["design"], r["split"], r["model"], r["policy"], r["device"], r["family"]) == k)
        mean = lambda f: np.nanmean([r[f] for r in rs])
        lines.append("|".join(map(str, k)) + f"|{len(rs)}|{nf}|{mean('inliers'):.0f}|{mean('out'):.1f}|{mean('out_photo'):.1f}|{mean('inz'):.1f}|{mean('hp_out'):.1f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
