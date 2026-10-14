"""DIAGNOSTIC: separate the two shifts behind the Protocol C failure, with the locked fold-0 checkpoints (no training).

Test 1, capture shift alone: SIDTD fold-0 test queries, native vs. simulated physical capture (blur, noise, colour cast,
illumination ramp, JPEG, small perspective) applied to the raw file; same locked banks and matrices.  If the AUC
collapses, capture texture alone defeats the checkpoints.

Test 2, attack mechanism alone: FantasyID support captures (reference-only genuine of the three C designs, never test
queries), genuine vs. SIDTD-style copy-and-replace forgeries made from them (a text field of another card of the same
design and device pasted into the same field), scored against homography banks built from the other identity half,
same-device where the half has >= 5 contributors, with photometric alignment.  If the AUC is high, the checkpoints read
SIDTD-style forgeries on physical captures; the C failure is then the attack mechanism, not the capture.
"""
import argparse, csv, json, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from refer_or_clear.data.rectify import to_canvas
from refer_or_clear.data.enrollment import _contributor
from refer_or_clear.data.register import _features
from refer_or_clear.data.direct_reference import _available
from refer_or_clear.models.direct_reference import build_direct_reference_model
from c_robust import Entries, build_bank, fit, overlap_of, read_mask, run, summarize, load_state, device_of, reencode, SEED

TEXT_FIELDS = ("name_en", "dob", "doi", "doe", "name", "surname", "number", "name_cn")


def capture_sim_raw(img, rng):
    x = img.astype(np.float32); h, w = x.shape[:2]; s = w / 800.
    gain = np.float32([rng.uniform(.85, 1.15) for _ in range(3)]); x = x * gain + rng.uniform(-12, 12)
    g = rng.uniform(-.25, .25); ramp = np.linspace(-.5, .5, w, dtype=np.float32)[None, :] if rng.random() < .5 else np.linspace(-.5, .5, h, dtype=np.float32)[:, None]
    x = x * (1 + g * ramp)[..., None]
    x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.3, 1.5) * s)
    x = x + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 8), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = src + np.float32([[rng.uniform(-3, 3) * s, rng.uniform(-3, 3) * s] for _ in range(4)])
    x = cv2.warpPerspective(x, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return reencode(x, rng.randint(60, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--raw-root", required=True); ap.add_argument("--a-eval-registration", required=True)
    ap.add_argument("--c-support", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--weights-sha256", required=True)
    ap.add_argument("--ckpt-on", required=True); ap.add_argument("--ckpt-off", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); raw_root = Path(a.raw_root); rng = random.Random(SEED)
    rec = {r["image_id"]: r for r in csv.DictReader(open(a.records))}

    # ---- Test 1 entries
    reg = json.load(open(a.a_eval_registration))["entries"]; sid_native, sid_sim = [], []
    simdir = out / "sidtd-sim"; simdir.mkdir(exist_ok=True)
    for k, v in reg.items():
        if v["role"] != "test": continue
        r = rec[k]; prod = {n: v["products"][n]["path"] for n in ("target", "target_valid", "valid", "zone", "median")}
        base = dict(image_id=k, label=int(r["label"]), family=r["attack_family"] if r["label"] == "1" else "genuine", dataset="sidtd", capture="template",
                    products=prod, matrix=v["query_canvas_to_bank"] if v["status"] == "fit" else None)
        sid_native.append(dict(base, path=v["path"]))
        p = simdir / f"{k}.jpg"
        if not p.exists():
            img = cv2.imread(str(raw_root / v["path"]), cv2.IMREAD_COLOR); cv2.imwrite(str(p), capture_sim_raw(img, rng), [cv2.IMWRITE_JPEG_QUALITY, 100])
        sid_sim.append(dict(base, path=str(p)))
    print("test1", len(sid_native), flush=True)

    # ---- Test 2 entries
    fakedir = out / "fantasy-copymove"; fakedir.mkdir(exist_ok=True); fan = []
    for design in sorted(p.name for p in (Path(a.c_support) / "fantasyid").iterdir()):
        man = json.load(open(Path(a.c_support) / "fantasyid" / design / "enrollment" / "manifest.json")); items = []
        for iid in man["support_ids"]:
            meta = man["contributors"][iid]; _, rects, _ = _contributor(raw_root, meta)
            img = cv2.imread(str(raw_root / meta["path"]), cv2.IMREAD_COLOR); canvas, valid, native = to_canvas(img)
            items.append(dict(image_id=iid, path=meta["path"], raw=img, canvas=canvas, valid=valid, native=native, feats=_features(canvas), rectangles=rects,
                              device=device_of(meta["path"]), identity=meta["identity_id"]))
        idents = sorted({it["identity"] for it in items}); half = {d: i % 2 for i, d in enumerate(idents)}
        for h in (0, 1):
            bank_items = [it for it in items if half[it["identity"]] == h]; probe = [it for it in items if half[it["identity"]] != h]
            banks = {"mixed": build_bank(bank_items, "homography", out / "banks" / design / f"half{h}" / "mixed", SEED)[0]}
            for dev in sorted({it["device"] for it in bank_items}):
                sub = [it for it in bank_items if it["device"] == dev]
                if len(sub) >= 5: banks[dev] = build_bank(sub, "homography", out / "banks" / design / f"half{h}" / dev, SEED)[0]
            tfe = {k: _features(cv2.imread(b["target"], cv2.IMREAD_COLOR)) for k, b in banks.items()}
            bm = {k: {n: read_mask(b[n]) for n in ("target_valid", "valid", "zone")} for k, b in banks.items()}
            for it in probe:
                bname = it["device"] if it["device"] in banks else "mixed"
                donors = [d for d in probe if d["device"] == it["device"] and d["identity"] != it["identity"]]
                variants = [("genuine", it["raw"], 0)]
                if donors:
                    donor = rng.choice(donors); fields = [r for r in it["rectangles"] if r[0] in TEXT_FIELDS and any(d[0] == r[0] for d in donor["rectangles"])]
                    if fields:
                        f = rng.choice(fields); df = next(d for d in donor["rectangles"] if d[0] == f[0])
                        _, x, y, w, hh = [int(round(v)) if i else v for i, v in enumerate(f)]; _, dx, dy, dw, dh = [int(round(v)) if i else v for i, v in enumerate(df)]
                        fake = it["raw"].copy(); crop = donor["raw"][dy:dy + dh, dx:dx + dw]
                        fake[y:y + hh, x:x + w] = cv2.resize(crop, (w, hh), interpolation=cv2.INTER_AREA)
                        variants.append((f"copy_{f[0]}", fake, 1))
                for name, img, lab in variants:
                    p = fakedir / f"{it['image_id']}__{name}.jpg"
                    if not p.exists(): cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    canvas, qvalid, _ = to_canvas(cv2.imread(str(p), cv2.IMREAD_COLOR))
                    M, inl = fit(_features(canvas), tfe[bname], "homography", SEED)
                    ok = M is not None and _available(overlap_of(qvalid, bm[bname], M)[1])
                    fan.append(dict(image_id=f"{it['image_id']}__{name}", path=str(p), label=lab, family="genuine" if lab == 0 else "text_copy", dataset="fantasyid-support",
                                    capture=it["device"], products=banks[bname], matrix=M.tolist() if ok else None, design=design))
    print("test2", len(fan), "genuine", sum(e["label"] == 0 for e in fan), "unregistered", sum(e["matrix"] is None for e in fan), flush=True)

    dev = torch.device("cuda"); result = {}
    for arm, ck in (("on", a.ckpt_on), ("off", a.ckpt_off)):
        ref_on = arm == "on"
        model = build_direct_reference_model(Path(a.weights), a.weights_sha256, reference_enabled=ref_on, head_seed=SEED, adapter_seed=SEED, local_enabled=ref_on, reference_bn="shared").to(dev)
        load_state(model, ck)
        for name, ents, align in (("sidtd_native", sid_native, True), ("sidtd_capture_sim", sid_sim, True), ("sidtd_capture_sim_noalign", sid_sim, False),
                                  ("fantasy_support_copymove", fan, True), ("fantasy_support_copymove_noalign", fan, False)):
            loader = DataLoader(Entries(ents, raw_root, ref_on, False, align=align), batch_size=8, shuffle=False, num_workers=a.workers)
            rows = run(model, loader, dev, ref_on); result[f"{arm}:{name}"] = summarize(rows)
            with (out / f"{arm}_{name}_predictions.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            s = result[f"{arm}:{name}"]
            print(arm, name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)}), flush=True)
            print("   ", json.dumps({k: (round(v["auc"], 3) if v["auc"] is not None else None) for k, v in s.items() if isinstance(v, dict)}), flush=True)
    (out / "summary.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
