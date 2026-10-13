"""Score the R reproduction (ResNet50, raw files resized to 299) on fold-0 validation, native and re-encoded at JPEG Q75."""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision
RUN = Path("/shared/b00090279/refer-or-clear/results/runs/anchor-pilots-v1/anchor-pilot-train-37967")
RAW = Path("/shared/b00090279/refer-or-clear/data/raw")
OUT = Path("/shared/b00090279/diag-align/r-reencoded"); OUT.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
records = {r["image_id"]: r for r in csv.DictReader(open(RUN / "inputs/records.csv"))}
val = [r for r in csv.DictReader(open(RUN / "inputs/split.csv")) if r["role"] == "val"]
assert len(val) == 222, len(val)
ck = torch.load(RUN / "pilot/checkpoints/best.pt", map_location="cpu", weights_only=False)
model = torchvision.models.resnet50(weights=None); model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict({k.removeprefix("backbone."): v for k, v in ck["state_dict"].items()})
model = model.cuda().eval()
def prep(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); rgb = cv2.resize(rgb, (299, 299), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(((rgb.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1)[None]).cuda()
def reencode(bgr, q=75):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q]); assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int); p, n = s[y == 1], s[y == 0]
    return float((p[:, None] > n[None, :]).mean() + .5 * (p[:, None] == n[None, :]).mean())
rows = []
with torch.inference_mode():
    for i, r in enumerate(val):
        rec = records[r["image_id"]]; bgr = cv2.imread(str(RAW / rec["path"]), cv2.IMREAD_COLOR); assert bgr is not None, rec["path"]
        out = {}
        for name, img in (("native", bgr), ("q75", reencode(bgr))):
            l = model(prep(img))[0]; out[name] = float(torch.sigmoid(l[1] - l[0]))
        rows.append(dict(image_id=r["image_id"], label=int(r["label"]), family=rec["attack_family"], **out))
        if i % 50 == 0: print(i, flush=True)
with (OUT / "scores.csv").open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
res = dict(checkpoint_epoch=ck["epoch"], n=len(rows))
gen = [r for r in rows if r["label"] == 0]
for name in ("native", "q75"):
    y = [r["label"] for r in rows]; s = [r[name] for r in rows]
    e = dict(accuracy=float(np.mean([(v > .5) == bool(l) for v, l in zip(s, y)])), auc=auc(s, y),
             genuine_fp=float(np.mean([r[name] > .5 for r in gen])))
    for fam in sorted({r["family"] for r in rows if r["label"] == 1}):
        sub = [r for r in rows if r["family"] == fam]
        e[fam] = dict(recall=float(np.mean([r[name] > .5 for r in sub])), auc=auc([r[name] for r in gen + sub], [r["label"] for r in gen + sub]))
    res[name] = e
(OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
