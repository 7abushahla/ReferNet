#!/usr/bin/env python3
"""Fixed-checkpoint fold-0 diagnosis. Test data only verifies the previously reported result."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear import evaluate as evaluation
from refer_or_clear import metrics
from refer_or_clear.data.dataset import PairDataset
from refer_or_clear.data.rectify import to_canvas
from diagnose_control import sha256, summarize
import score_sidtd_checkpoints as published


class ProbeDataset(PairDataset):
    def __init__(self, pairs, mode):
        self.mode = mode
        super().__init__(pairs, train=mode != "native", crop=None, jpeg=None,
                         affine_px=8, jitter_px=2 if mode in ("full", "jitter") else 0,
                         photometric=mode in ("full", "photometric"))

    def _geometry(self, height, width):
        if self.mode in ("geometry", "full"):
            return super()._geometry(height, width)
        return np.float32([[1, 0, 0], [0, 1, 0]]), (slice(0, height), slice(0, width))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows):
    from sklearn.metrics import roc_auc_score

    groups = {"all": rows}
    for family in sorted({row["attack_family"] for row in rows if row["label"] == 1}):
        groups[family] = [row for row in rows if row["label"] == 0 or row["attack_family"] == family]
    result = {}
    for name, selected in groups.items():
        labels = [row["label"] for row in selected]
        block = summarize(labels, [row["logit_fake"] for row in selected])
        block["map_peak_auc"] = float(roc_auc_score(labels, [row["map_peak"] for row in selected])) if len(set(labels)) == 2 else None
        block["empty_forged_masks"] = sum(row["label"] == 1 and row["mask_pixels"] == 0 for row in selected)
        result[name] = block
    return result


@torch.no_grad()
def score_ours(model, pairs, mode, batch_size, device):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = DataLoader(ProbeDataset(pairs, mode), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    for batch in loader:
        output = model(batch["x"].to(device))
        logits = output["logit"].float().reshape(-1).cpu()
        peaks = output["map"].float().flatten(1).amax(1).cpu()
        masks = (batch["mask"] * batch["valid"]).flatten(1).sum(1)
        for index, logit in enumerate(logits.tolist()):
            rows.append(dict(image_id=batch["meta"]["image_id"][index],
                             design_id=batch["meta"]["design_id"][index],
                             attack_family=batch["meta"]["attack_family"][index],
                             label=int(batch["label"][index]), logit_fake=logit,
                             map_peak=float(peaks[index]), mask_pixels=int(masks[index])))
    assert [row["image_id"] for row in rows] == [pair["image_id"] for pair in pairs]
    return rows


@torch.no_grad()
def score_published(model, pairs, records, raw_root, mode, device):
    rows = []
    for pair in pairs:
        raw_path = raw_root / records.at[pair["image_id"], "path"]
        source = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(raw_path)
        if mode == "png_roundtrip":
            ok, encoded = cv2.imencode(".png", source)
            assert ok
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            assert np.array_equal(source, decoded)
            source = decoded
        elif mode == "resize_canvas":
            source = to_canvas(source)[0]
        elif mode == "aligned_canvas":
            source = cv2.imread(pair["path"], cv2.IMREAD_COLOR)
        inputs = torch.from_numpy(published.preprocess(source, 299)).unsqueeze(0).to(device)
        logits = model(inputs).float()[0].cpu()
        mask = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE) if pair["mask_path"] else None
        rows.append(dict(image_id=pair["image_id"], design_id=pair["design_id"],
                         attack_family=pair["attack_family"], label=pair["label"],
                         logit_fake=float(logits[1] - logits[0]), map_peak=0.0,
                         mask_pixels=int((mask > 127).sum()) if mask is not None else 0))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="A-cex-single_noex-median-levir-f0-s42-bnbatch-e100")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    info = json.loads((Path("results/runs") / args.run / "train.json").read_text())
    expected = json.loads((Path("results/runs") / args.run / "metrics.json").read_text())
    config = info["raw_config"]
    split, records = evaluation.load_split(config, args.data_root, fold=0)
    pairs = {role: evaluation.role_pairs(split, records, args.data_root, role, "median", config, 42,
                                         wrong_design_fraction=0.0) for role in ("train", "val", "test")}
    assert all(not pair["wrong_design"] for selected in pairs.values() for pair in selected)
    ids = {role: {pair["image_id"] for pair in selected} for role, selected in pairs.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    raw_root = Path(args.raw_root)
    official = pd.read_csv(raw_root / "sidtd/kfold_split_templates/split_kfold/val_split_SIDTD_it_0.csv")
    expected_val = {("sidtd_" + Path(row.image_path).stem, int(row.label)) for row in official.itertuples()}
    assert expected_val == {(pair["image_id"], pair["label"]) for pair in pairs["val"]}
    summary = dict(run=args.run, inputs={role: len(selected) for role, selected in pairs.items()},
                   script_sha256=sha256(__file__), torch_version=torch.__version__,
                   split_sha256=sha256(Path(args.data_root) / "splits/protocol_a_it0.csv"),
                   test_usage="One unchanged fresh-load verification only; diagnostic conditions use train/val.",
                   checkpoints={}, ours={}, published={})
    dump(out / "train_metadata.json", info)
    for checkpoint_name in ("best", "last"):
        path = Path("checkpoints") / args.run / f"{checkpoint_name}.pt"
        digest = sha256(path)
        model, checkpoint = evaluation.load_checkpoint(path, device)
        assert checkpoint["config"]["bn_policy"] == "batch_stats"
        assert checkpoint["config"]["fold"] == 0
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        summary["checkpoints"][checkpoint_name] = dict(sha256=digest, epoch=checkpoint["epoch"])
        if checkpoint_name == "best":
            predictions = evaluation.predict(model, evaluation.eval_loader(pairs["test"], 1, 2), device)
            actual = dict(auc=metrics.auc(predictions["score"], predictions["label"]),
                          accuracy=metrics.deepid_acc(predictions["score"], predictions["label"]))
            assert abs(actual["auc"] - expected["test"]["native"]["auc"]) < 1e-7
            assert abs(actual["accuracy"] - expected["test"]["native"]["deepid_acc"]) < 1e-7
            summary["fresh_load_verification"] = actual
            print("Fresh checkpoint verification passed", actual, flush=True)
        summary["ours"][checkpoint_name] = {}
        conditions = [(role, "native", 1) for role in ("train", "val")]
        conditions += [("val", "native", 8)]
        if checkpoint_name == "best":
            conditions += [("val", mode, 1) for mode in ("geometry", "photometric", "jitter", "full")]
        for role, mode, batch_size in conditions:
            key = f"{role}_{mode}_batch{batch_size}"
            rows = score_ours(model, pairs[role], mode, batch_size, device)
            summary["ours"][checkpoint_name][key] = group_summary(rows)
            write_rows(out / f"ours_{checkpoint_name}_{key}.csv", rows)
            print(checkpoint_name, key, {name: round(block["auc_logit"], 5) for name, block in summary["ours"][checkpoint_name][key].items()}, flush=True)
            dump(out / "summary.json", summary)
        assert all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
        assert sha256(path) == digest
        del model, state
    for name in ("resnet50", "efficientnet-b3"):
        checkpoint_path = published.pick_checkpoint(sorted((raw_root / "sidtd/paper_checkpoints" / name).rglob("*.pth")), 0)
        state = torch.load(checkpoint_path, map_location="cpu")
        model = published.build(name).to(device).eval()
        model.load_state_dict(state.get("state_dict", state))
        summary["published"][name] = {}
        original_rows = None
        for mode in ("raw", "png_roundtrip", "resize_canvas", "aligned_canvas"):
            rows = score_published(model, pairs["val"], records.set_index("image_id"), raw_root, mode, device)
            if mode == "raw":
                original_rows = rows
            elif mode == "png_roundtrip":
                assert rows == original_rows
            summary["published"][name][mode] = group_summary(rows)
            write_rows(out / f"{name}_{mode}_val.csv", rows)
            print(name, mode, {family: round(block["auc_logit"], 5) for family, block in summary["published"][name][mode].items()}, flush=True)
            dump(out / "summary.json", summary)
        summary["checkpoints"][name] = dict(sha256=sha256(checkpoint_path), path=checkpoint_path)
        del model, state
    summary["complete"] = True
    dump(out / "summary.json", summary)


if __name__ == "__main__":
    main()
