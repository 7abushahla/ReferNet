"""Protocol C candidate: zero-shot transfer of the SIDTD fold checkpoints to an IDNet design (e.g. Arizona driver licence):
unseen design, unseen generator, same digital image formation, with target genuine enrollment and no target training.

Support: n genuine images of the design (sorted stems, seeded shuffle) build the enrollment bank with the study's own
similarity registration and two-pass median; the personal-data zone is the union of the annotated field boxes (text fields,
face and ghost image).  Queries: genuine images and attacks of the remaining identities (attack files derived from, or
donating to, a support identity are excluded).  Families: inpaint_and_rewrite (fraud5), crop_and_replace (fraud6),
text_replace (fraud1_copy_and_move, in-place rewrite), face_replace (fraud3), face_morph (fraud2).
Scored native and re-encoded at IJG quality 75 (genuine files are PNG, fraud5/6 JPEG: the encoding control is mandatory).
"""
import argparse, csv, json, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from refer_or_clear.data.rectify import to_canvas
from refer_or_clear.data.register import _features
from refer_or_clear.data.direct_reference import _available
from refer_or_clear.models.direct_reference import build_direct_reference_model
from c_robust import Entries, build_bank, fit, overlap_of, read_mask, run, summarize, load_state, SEED

FAMILIES = {"fraud5_inpaint_and_rewrite": "text_inpaint", "fraud6_crop_and_replace": "text_copy", "fraud1_copy_and_move": "text_replace",
            "fraud3_face_replacement": "face_replace", "fraud2_face_morphing": "face_morph"}


def stem_of(name):
    s = Path(name).stem
    return s.split("_fake_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-root", required=True); ap.add_argument("--design", required=True); ap.add_argument("--runs-root", required=True)
    ap.add_argument("--weights", required=True); ap.add_argument("--weights-sha256", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="0"); ap.add_argument("--arms", default="off,on"); ap.add_argument("--n-support", type=int, default=100)
    ap.add_argument("--n-per-class", type=int, default=300); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--align", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.design_root), Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = random.Random(SEED)
    meta = root / "meta" / "detailed_with_fraud_info"
    orig = json.load(open(meta / f"{a.design}_original_annotation.json"))
    crop = json.load(open(meta / f"{a.design}_crop_and_replace.json"))
    genuine = sorted(p.name for p in (root / "positive").iterdir() if p.name in orig)
    rng.shuffle(genuine); support_names = sorted(genuine[: a.n_support]); support = {stem_of(n) for n in support_names}
    rest = [n for n in genuine[a.n_support:]]

    def rects(name):
        return [[k, v["bbox"][0], v["bbox"][1], v["bbox"][2] - v["bbox"][0], v["bbox"][3] - v["bbox"][1]] for k, v in orig[name].items() if isinstance(v, dict) and "bbox" in v]

    items = []
    for n in support_names:
        img = cv2.imread(str(root / "positive" / n), cv2.IMREAD_COLOR); canvas, valid, native = to_canvas(img)
        items.append(dict(canvas=canvas, valid=valid, native=native, feats=_features(canvas), rectangles=rects(n)))
    bank, used = build_bank(items, "similarity", out / "bank", SEED)
    print(f"bank {a.design}: {len(items)} support, registered {used}", flush=True)
    tf = _features(cv2.imread(bank["target"], cv2.IMREAD_COLOR)); bm = {n: read_mask(bank[n]) for n in ("target_valid", "valid", "zone")}

    # ---- queries
    entries = []
    q_gen = sorted(rest); rng.shuffle(q_gen); q_gen = q_gen[: a.n_per_class]
    for n in q_gen:
        entries.append(dict(image_id=f"{a.design}_genuine_{Path(n).stem}", path=f"positive/{n}", label=0, family="genuine"))
    for folder, fam in FAMILIES.items():
        files = sorted(p.name for p in (root / folder).iterdir())
        ok = []
        for f in files:
            s = stem_of(f)
            if s in support: continue
            if folder == "fraud6_crop_and_replace":
                ann = crop.get(Path(f).stem)
                if ann and (stem_of(ann["src"]["name"]) in support or stem_of(ann["des"]["name"]) in support): continue
            ok.append(f)
        rng.shuffle(ok)
        for f in ok[: a.n_per_class]:
            entries.append(dict(image_id=f"{a.design}_{fam}_{Path(f).stem}", path=f"{folder}/{f}", label=1, family=fam))
    for e in entries:
        canvas, qvalid, _ = to_canvas(cv2.imread(str(root / e["path"]), cv2.IMREAD_COLOR))
        M, inl = fit(_features(canvas), tf, "similarity", SEED)
        okk = M is not None and _available(overlap_of(qvalid, bm, M)[1])
        e.update(matrix=M.tolist() if okk else None, inliers=inl, products=bank, dataset=f"idnet-{a.design}", capture="render")
    print("queries", len(entries), "genuine", sum(e["label"] == 0 for e in entries), "unregistered", sum(e["matrix"] is None for e in entries), flush=True)
    (out / "queries.json").write_text(json.dumps([{k: v for k, v in e.items() if k != "products"} for e in entries]))

    dev = torch.device("cuda"); result = {}
    folds = [int(x) for x in (range(int(a.folds.split("-")[0]), int(a.folds.split("-")[1]) + 1) if "-" in a.folds else a.folds.split(","))]
    for fold in folds:
        for arm in a.arms.split(","):
            ref_on = arm == "on"; ck = Path(a.runs_root) / f"fold{fold}-v2noaux-fdb7351" / "training" / arm / "best.pt"
            model = build_direct_reference_model(Path(a.weights), a.weights_sha256, reference_enabled=ref_on, head_seed=SEED, adapter_seed=SEED, local_enabled=ref_on, reference_bn="shared").to(dev)
            ep = load_state(model, str(ck))
            for variant, q in (("native", None), ("q75", 75)):
                loader = DataLoader(Entries(entries, root, ref_on, False, quality=q, align=a.align), batch_size=8, shuffle=False, num_workers=a.workers)
                rows = run(model, loader, dev, ref_on); key = f"fold{fold}:{arm}:{variant}"; result[key] = dict(summarize(rows), epoch=ep)
                with (out / f"{key.replace(':', '_')}_predictions.csv").open("w", newline="") as h:
                    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                s = result[key]
                print(key, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)}), flush=True)
                print("   ", json.dumps({k: (round(v["auc"], 3) if v["auc"] is not None else None) for k, v in s.items() if isinstance(v, dict)}), flush=True)
            (out / "summary.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    # ---- fold aggregate
    agg = {}
    for arm in a.arms.split(","):
        for variant in ("native", "q75"):
            ks = [f"fold{f}:{arm}:{variant}" for f in folds if f"fold{f}:{arm}:{variant}" in result]
            if not ks: continue
            m = lambda f: (float(np.mean([f(result[k]) for k in ks])), float(np.std([f(result[k]) for k in ks], ddof=1)) if len(ks) > 1 else 0.0)
            agg[f"{arm}:{variant}"] = dict(n_folds=len(ks), accuracy=m(lambda s: s["accuracy"]), auc=m(lambda s: s["auc"]),
                                           **{fam: m(lambda s, fam=fam: s[f"family:{fam}"]["auc"]) for fam in FAMILIES.values() if f"family:{fam}" in result[ks[0]]})
    (out / "aggregate.json").write_text(json.dumps(agg, indent=1, sort_keys=True)); print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
