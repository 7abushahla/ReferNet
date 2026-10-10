# SIDTD Protocols A (anchor) and B (held-out nationalities) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run our three arms under the SIDTD paper's template 10-fold protocol (Protocol A, the anchor), check the control against the published numbers, then run the same arms under the paper's 10 held-out-nationality groups (Protocol B).

**Architecture:** Canvases and references become lossless PNG, so no image is re-encoded after the raw file. Split builders turn the paper's fold files and nationality groups into one split CSV per fold or group, and Protocol A gets per-fold design medians built from that fold's training files only. Training gains a `--fold` argument, the paper's checkpoint rule (best validation accuracy, AUC breaks ties), JPEG augmentation off, and no wrong-design pairs for the control; the sweep generator and the aggregator learn folds, and the aggregator prints the Table 5 layout and the gate.

**Tech Stack:** Python 3.10, PyTorch, OpenCV, pandas, PyYAML, scikit-learn; Open-CD ChangerEx on the AWS SLURM cluster (account acc-mialhajri).

**Spec:** `docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md`

## Global Constraints

- Tests are plain-assert scripts run as `$PY tests/<file>.py`; there is no pytest in this repository. Each test file ends with a `__main__` block that calls every test and prints `ok`.
- Local interpreter: `PY=/Users/hamza/Documents/GitHub/Thesis/SANA-FE/.venv/bin/python` (torch, cv2, pandas, yaml, sklearn; no mmseg or Open-CD, so ChangerEx itself runs only on the cluster). All local commands run from `~/Documents/GitHub/refer-or-clear`.
- Canvas: 512 x 800 (`CANVAS_H, CANVAS_W` in `src/refer_or_clear/data/rectify.py`), unchanged, for every dataset.
- No re-encoding after the raw file: canvases and references are PNG; `augment.jpeg_quality: null` in Protocols A and B.
- Splits are used as given: `data/raw/sidtd/kfold_split_templates/split_kfold/{train,val,test}_split_SIDTD_it_{k}.csv`, k = 0..9; the ten nationality groups verbatim from `SIDTD/models/fsl_models/test_fsl.py`: svk_alb_aze_lva, svk_rus_esp_aze, grc_esp_fin_srb, svk_alb_grc_est, alb_grc_rus_lva, alb_grc_est_rus, svk_rus_lva_fin, rus_aze_lva_fin, svk_alb_lva_fin, svk_alb_rus_fin.
- Arms (arch `changerex_r18`, init `levir`, seed 42): control = fusion `single` with reference `median`; median = fusion `difference`, reference `median`; specimen = fusion `difference`, reference `specimen`.
- Checkpoint: best validation accuracy at 0.5, ties broken by validation ROC AUC, earliest epoch on a full tie.
- Gate: the control's mean ROC AUC over the 10 folds is at least 0.99 and its mean accuracy at least 0.97, with all 10 folds present. Protocol B is not submitted before the gate passes and the user has seen the Protocol A table.
- Cluster jobs: 4 CPUs and 8G per job (cluster caps), submitted from the repository root on the login node.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Uneven folds: fold 4 has 1,776 / 223 / 223 and fold 9 has 1,777 / 222 / 223 images; nothing may assume 222 test images. Pinned in Task 2.
2. A design whose fold training file leaves one hash half without genuine images: the per-fold median must fail loudly, never fall back to the global median. Pinned in Task 2.
3. Stale JPEG canvases or references from the old pipeline left in `data/processed`: pairs must resolve only `.png` and raise when it is missing. Pinned in Task 1.
4. Superseded runs whose tags carry no fold (`A-cex-single-median-levir-s42`) must not enter Protocol A or B tables, and must not be mistaken for a finished fold by the sweep skip rule. Pinned in Tasks 6 and 7.
5. The control must never receive wrong-design pairs, in training or through the evaluator CLI. Pinned in Task 5.

---

### Task 1: Lossless canvases and references

**Files:**
- Modify: `src/refer_or_clear/data/rectify.py` (add `CANVAS_EXT`)
- Modify: `scripts/prepare_data.py` (`_canvas_job`, canvas paths, stale-JPEG cleanup, docstring)
- Modify: `src/refer_or_clear/data/register.py:72`
- Modify: `src/refer_or_clear/data/references.py` (median inputs and outputs, `reference_path`, `build_specimens`)
- Modify: `src/refer_or_clear/data/pairs.py` (canvas and reference paths)
- Modify: `scripts/show_data.py` (reference and canvas paths)
- Test: `tests/test_lossless_canvas.py` (create)

**Interfaces:**
- Produces: `refer_or_clear.data.rectify.CANVAS_EXT = ".png"`; canvases at `canvases/<image_id>.png`; references at `references/<design_tag>_fold{0,1}.png`, `<design_tag>_all.png`, `<design_tag>_specimen_fold{0,1}.png`, `<design_tag>_specimen.png`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lossless_canvas.py`:

```python
"""Canvases and references are lossless: python tests/test_lossless_canvas.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from refer_or_clear.data.rectify import CANVAS_EXT, to_canvas  # noqa: E402
import prepare_data as PD  # noqa: E402
from refer_or_clear.data import pairs as P  # noqa: E402


def test_extension():
    assert CANVAS_EXT == ".png", CANVAS_EXT


def test_canvas_job_is_lossless():
    img = np.random.default_rng(0).integers(0, 256, (400, 640, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "src.png"
        cv2.imwrite(str(src), img)
        dst = d / f"q{CANVAS_EXT}"
        PD._canvas_job(("q", src, dst, d / "v.png", None))
        expected, _, _ = to_canvas(img)
        assert np.array_equal(cv2.imread(str(dst)), expected), "canvas changed on write"


def test_pairs_reject_stale_jpeg():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "canvases").mkdir()
        cv2.imwrite(str(d / "canvases" / "x.jpg"), np.zeros((4, 4, 3), np.uint8))
        try:
            P._require(d / "canvases" / f"x{CANVAS_EXT}")
        except FileNotFoundError:
            return
        raise AssertionError("a stale .jpg canvas was accepted")


if __name__ == "__main__":
    test_extension()
    test_canvas_job_is_lossless()
    test_pairs_reject_stale_jpeg()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_lossless_canvas.py`
Expected: `ImportError: cannot import name 'CANVAS_EXT'`

- [ ] **Step 3: Add the constant**

In `src/refer_or_clear/data/rectify.py`, directly below `CANVAS_H, CANVAS_W = 512, 800` (line 14), add:

```python
CANVAS_EXT = ".png"  # lossless: nothing is re-encoded after the raw file (spec of 10 Oct 2026)
```

- [ ] **Step 4: Write canvases losslessly**

In `scripts/prepare_data.py`:
1. Change the import line `from refer_or_clear.data.rectify import to_canvas  # noqa: E402` to
   `from refer_or_clear.data.rectify import CANVAS_EXT, to_canvas  # noqa: E402`.
2. In `_canvas_job`, rename `dst_jpg` to `dst_img` and replace
   `cv2.imwrite(str(dst_jpg), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])` with
   `cv2.imwrite(str(dst_img), canvas)`.
3. Replace each of the three occurrences of `out / "canvases" / f"{r.image_id}.jpg"` with
   `out / "canvases" / f"{r.image_id}{CANVAS_EXT}"`.
4. At the start of `main()`, directly after the `for d in (...): mkdir` loop, add:

```python
    stale_jpg = list((out / "canvases").glob("*.jpg")) + list((out / "references").glob("*.jpg"))
    if stale_jpg:
        print(f"      removing {len(stale_jpg)} JPEG canvases and references from the old pipeline")
        for p in stale_jpg:
            p.unlink()
        a.force = True
```

5. In the module docstring, change `canvases/<image_id>.jpg` to `canvases/<image_id>.png` and `references/<design>_{fold0,fold1,all}.jpg` to `references/<design>_{fold0,fold1,all}.png`.

- [ ] **Step 5: Read and write PNG everywhere else**

- `src/refer_or_clear/data/register.py:72`: add `from .rectify import CANVAS_EXT` to the imports and change `canvas_dir / f"{i}.jpg"` to `canvas_dir / f"{i}{CANVAS_EXT}"`.
- `src/refer_or_clear/data/references.py`: change the import to `from .rectify import CANVAS_EXT, CANVAS_H, CANVAS_W, rect_to_canvas`. In `build_references`, replace both `canvas_dir / f"{i}.jpg"` with `canvas_dir / f"{i}{CANVAS_EXT}"`, replace `cv2.imwrite(str(out_dir / f"{tag}_fold{fold}.jpg"), med, [cv2.IMWRITE_JPEG_QUALITY, 95])` with `cv2.imwrite(str(out_dir / f"{tag}_fold{fold}.png"), med)` and `cv2.imwrite(str(out_dir / f"{tag}_all.jpg"), med, [cv2.IMWRITE_JPEG_QUALITY, 95])` with `cv2.imwrite(str(out_dir / f"{tag}_all.png"), med)`. In `reference_path`, change both `.jpg` suffixes to `.png`. In `build_specimens`, change `f"{tag}_all.jpg"` (twice), `f"{tag}_fold{fold}.jpg"`, `f"{tag}_specimen_fold{fold}.jpg"` and `f"{tag}_specimen.jpg"` to `.png`, and replace `cv2.imwrite(str(out), spec, [cv2.IMWRITE_JPEG_QUALITY, 95])` with `cv2.imwrite(str(out), spec)`. Leave `templates_dir / f"{stem}.jpg"` unchanged: that is the raw specimen file. In the comment block above `SPECIMEN_DESIGNS`, change `<design>_specimen.jpg` to `<design>_specimen.png`.
- `src/refer_or_clear/data/pairs.py`: add `from .rectify import CANVAS_EXT`; change `f"{tag}_fold{1 - fold_of(component_id)}.jpg"` and `f"{tag}_specimen_fold{1 - fold_of(component_id)}.jpg"` to `.png`; change both `self.root / "canvases" / f"{...}.jpg"` and `root / "canvases" / f"{r.image_id}.jpg"` to use `{CANVAS_EXT}`.
- `scripts/show_data.py`: change `_all.jpg` (lines 40 and 46) to `_all.png` and `f"{r.image_id}.jpg"` (line 61) to `f"{r.image_id}.png"`. The output figures stay JPEG.

- [ ] **Step 6: Run the test and the existing checks**

Run: `$PY tests/test_lossless_canvas.py && $PY tests/test_metrics.py`
Expected: `ok` from the first and the existing pass output from the second. Then run `grep -rn '\.jpg' src scripts --include=*.py | grep -v audit_jpeg` and confirm the only remaining hits are raw-file names in `records.py`, `forgeries.py`, `references.py` (`{nn}.jpg` annotation keys and `{stem}.jpg` specimens), `dataset.py` (`imencode(".jpg")` for the equalized variant) and the figure outputs of `show_data.py`.

- [ ] **Step 7: Commit**

```bash
git add src/refer_or_clear/data/rectify.py src/refer_or_clear/data/register.py src/refer_or_clear/data/references.py \
        src/refer_or_clear/data/pairs.py scripts/prepare_data.py scripts/show_data.py tests/test_lossless_canvas.py
git commit -m "Lossless PNG canvases and references; no re-encoding after the raw file

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Protocol A splits (the paper's 10 folds) and per-fold references

**Files:**
- Modify: `src/refer_or_clear/data/splits.py` (add `protocol_a_kfold`, `REF_COL`; remove the old `protocol_a` and `_sidtd_official_fewshot`)
- Modify: `src/refer_or_clear/data/references.py` (`build_references` gains `group` and `write_all`)
- Modify: `src/refer_or_clear/data/pairs.py` (`_RefPicker.__call__` gains `group`; `build_pairs` reads `ref_group`)
- Modify: `scripts/prepare_data.py` (step 7 writes the fold CSVs and builds the fold references)
- Test: `tests/test_splits_anchor.py` (create)

**Interfaces:**
- Consumes: `CANVAS_EXT` (Task 1).
- Produces:
  - `splits.REF_COL = "ref_group"`; `splits.protocol_a_kfold(records: pd.DataFrame, root: Path) -> dict[int, pd.DataFrame]` with columns `COLS + [REF_COL]`, `ref_group == f"it{k}"`.
  - `references.build_references(records, canvas_dir, out_dir, group: str = "", write_all: bool = True) -> pd.DataFrame`; group files are `<design_tag>_<group>_fold{0,1}.png`.
  - `pairs._RefPicker.__call__(design_id: str, component_id: str, group: str = "") -> str`.
  - Files `data/processed/splits/protocol_a_it{k}.csv`, k = 0..9.

- [ ] **Step 1: Write the failing test**

Create `tests/test_splits_anchor.py`:

```python
"""Protocol A folds and per-fold references: python tests/test_splits_anchor.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear.data import pairs as P  # noqa: E402
from refer_or_clear.data.references import build_references, fold_of  # noqa: E402
from refer_or_clear.data.splits import REF_COL, protocol_a_kfold  # noqa: E402

RAW = ROOT / "data" / "raw"
RECORDS = pd.read_csv(ROOT / "data" / "processed" / "records.csv", keep_default_na=False)
KDIR = RAW / "sidtd" / "kfold_split_templates" / "split_kfold"


def test_folds_match_the_paper_files():
    folds = protocol_a_kfold(RECORDS, RAW)
    assert sorted(folds) == list(range(10))
    n_sidtd = int((RECORDS.dataset == "sidtd").sum())
    tests = []
    for k, df in folds.items():
        assert len(df) == n_sidtd, (k, len(df))
        assert set(df[REF_COL]) == {f"it{k}"}
        for role in ("train", "val", "test"):
            n_csv = len(pd.read_csv(KDIR / f"{role}_split_SIDTD_it_{k}.csv"))
            assert int((df.role == role).sum()) == n_csv, (k, role)
        tests += list(df[df.role == "test"].image_id)
    assert len(tests) == len(set(tests)) == n_sidtd, "every image is tested exactly once"
    assert (folds[4].role == "train").sum() == 1776 and (folds[9].role == "test").sum() == 223


def _toy_canvases(d: Path, comps: list[str]) -> pd.DataFrame:
    (d / "canvases").mkdir()
    rows = []
    for i, c in enumerate(comps):
        cv2.imwrite(str(d / "canvases" / f"g{i}.png"), np.full((8, 8, 3), 10 * (i + 1), np.uint8))
        rows.append(dict(image_id=f"g{i}", design_id="x/d", identity_id=c, component_id=c, label=0))
    return pd.DataFrame(rows)


def test_group_references_use_only_the_given_records():
    comps = [f"c{i}" for i in range(12)]
    assert {fold_of(c) for c in comps} == {0, 1}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = _toy_canvases(d, comps)
        sub = rec[rec.component_id.isin(comps[:8])]
        build_references(sub, d / "canvases", d / "references", group="it3", write_all=False)
        names = sorted(p.name for p in (d / "references").glob("*.png"))
        assert "x_d_it3_fold0.png" in names and "x_d_it3_fold1.png" in names, names
        assert not any("_all" in n for n in names), names
        half = [i for i, c in enumerate(comps[:8]) if fold_of(c) == 0]
        expected = np.median([10 * (i + 1) for i in half])
        got = cv2.imread(str(d / "references" / "x_d_it3_fold0.png"))[0, 0, 0]
        assert abs(int(got) - int(expected)) <= 1, (got, expected)


def test_group_reference_fails_loudly_on_an_empty_half():
    comps = [c for c in (f"c{i}" for i in range(40)) if fold_of(c) == 0][:3]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec = _toy_canvases(d, comps)
        try:
            build_references(rec, d / "canvases", d / "references", group="it0", write_all=False)
        except ValueError as e:
            assert "x/d" in str(e) and "it0" in str(e)
            return
        raise AssertionError("an empty half produced no error")


def test_picker_resolves_group_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "references").mkdir()
        for name in ("x_d_it2_fold0.png", "x_d_it2_fold1.png", "x_d_fold0.png", "x_d_fold1.png"):
            (d / "references" / name).write_bytes(b"0")
        rec = pd.DataFrame([dict(image_id="g", design_id="x/d", component_id="c1", label=0)])
        pick = P._RefPicker(rec, d, "median", np.random.default_rng(0))
        other = 1 - fold_of("c1")
        assert pick("x/d", "c1", "it2").endswith(f"x_d_it2_fold{other}.png")
        assert pick("x/d", "c1").endswith(f"x_d_fold{other}.png")


if __name__ == "__main__":
    test_folds_match_the_paper_files()
    test_group_references_use_only_the_given_records()
    test_group_reference_fails_loudly_on_an_empty_half()
    test_picker_resolves_group_files()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_splits_anchor.py`
Expected: `ImportError: cannot import name 'REF_COL'`

- [ ] **Step 3: Implement the fold splits**

In `src/refer_or_clear/data/splits.py`, replace the module docstring's Protocol A paragraph with:

```
Protocol A (anchor): the SIDTD paper's ten official template folds (kfold_split_templates), used as given,
one split per fold; each row carries ref_group 'it<k>' so the design medians of fold k are built from that
fold's training file only.
```

Delete `_sidtd_official_fewshot` and `protocol_a`. Add after `_train_val`:

```python
REF_COL = "ref_group"
KFOLD_DIR = ("sidtd", "kfold_split_templates", "split_kfold")
N_FOLDS = 10


def _csv_ids(csv: Path) -> set[str]:
    p = pd.read_csv(csv)
    col = [c for c in p.columns if "path" in c.lower()][0]
    return {"sidtd_" + Path(x).stem for x in p[col]}


def protocol_a_kfold(records: pd.DataFrame, root: Path) -> dict[int, pd.DataFrame]:
    """Protocol A (anchor): fold k -> split frame, roles read from the paper's CSVs and never re-drawn."""
    d = Path(root).joinpath(*KFOLD_DIR)
    s0 = records[records.dataset == "sidtd"]
    known = set(s0.image_id)
    out = {}
    for k in range(N_FOLDS):
        ids = {role: _csv_ids(d / f"{role}_split_SIDTD_it_{k}.csv") for role in ("train", "val", "test")}
        for role, s in ids.items():
            if s - known:
                raise ValueError(f"fold {k} {role}: {len(s - known)} ids not in records, e.g. {sorted(s - known)[:3]}")
        if (ids["train"] & ids["val"]) or (ids["train"] & ids["test"]) or (ids["val"] & ids["test"]):
            raise ValueError(f"fold {k}: an image appears in two roles")
        s = s0.copy()
        s["protocol"], s["subset"], s[REF_COL] = "A", "", f"it{k}"
        s["role"] = ""
        for role, members in ids.items():
            s.loc[s.image_id.isin(members), "role"] = role
        if (s.role == "").any():
            raise ValueError(f"fold {k}: {int((s.role == '').sum())} SIDTD images are in no role")
        out[k] = s[COLS + [REF_COL]].reset_index(drop=True)
    return out
```

- [ ] **Step 4: Implement group references**

In `src/refer_or_clear/data/references.py`, replace `build_references` with:

```python
def build_references(records: pd.DataFrame, canvas_dir: Path, out_dir: Path, group: str = "",
                     write_all: bool = True) -> pd.DataFrame:
    """Fold medians per design from the genuine rows of `records`. With `group`, files are named
    <design>_<group>_fold{0,1}.png and both halves must be non-empty (no fallback to the global median)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    genuine = records[records.label == 0].copy()
    genuine["fold"] = genuine.component_id.map(fold_of)
    valid_dir = canvas_dir.parent / "valid"
    rows = []
    for design, g in genuine.groupby("design_id"):
        tag = design.replace("/", "_") + (f"_{group}" if group else "")
        for fold in (0, 1):
            sub = g[g.fold == fold]
            if not len(sub):
                if group:
                    raise ValueError(f"design {design}, group {group}: no genuine image in half {fold}")
                continue
            med, cover = median_canvas([canvas_dir / f"{i}{CANVAS_EXT}" for i in sub.image_id], valid_dir)
            cv2.imwrite(str(out_dir / f"{tag}_fold{fold}.png"), med)
            cv2.imwrite(str(out_dir / f"{tag}_fold{fold}_coverage.png"), cover)
        if write_all:
            med, cover = median_canvas([canvas_dir / f"{i}{CANVAS_EXT}" for i in g.image_id], valid_dir)
            cv2.imwrite(str(out_dir / f"{tag}_all.png"), med)
        rows.append(dict(design_id=design, group=group, n_genuine=len(g), n_identities=g.identity_id.nunique(),
                         n_components=g.component_id.nunique(), n_fold0=int((g.fold == 0).sum()),
                         n_fold1=int((g.fold == 1).sum()), min_coverage=int(cover.min())))
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Make pairs group-aware**

In `src/refer_or_clear/data/pairs.py`, replace `_RefPicker.__call__` with:

```python
    def __call__(self, design_id: str, component_id: str, group: str = "") -> str:
        tag, ref_dir = _tag(design_id), self.root / "references"
        if self.reference == "median":
            g = f"_{group}" if group else ""
            return _require(ref_dir / f"{tag}{g}_fold{1 - fold_of(component_id)}.png")
        if self.reference == "specimen":          # registered to the fold median the query never entered
            return _require(ref_dir / f"{tag}_specimen_fold{1 - fold_of(component_id)}.png")
        pool = sorted(i for i, c in self.genuine.get(design_id, []) if c != component_id)
        if not pool:
            raise FileNotFoundError(f"no genuine canvas of {design_id} outside component {component_id}")
        return _require(self.root / "canvases" / f"{pool[self.rng.integers(len(pool))]}{CANVAS_EXT}")
```

In `build_pairs`, inside the loop, add `group = _str(getattr(r, "ref_group", ""))` after `comp, label = ...`, change `ref_path=pick(r.design_id, comp)` to `ref_path=pick(r.design_id, comp, group)`, and add `ref_group=group` to the dict. In `_wrong_design`, change `ref_path=pick(other, p["component_id"])` to `ref_path=pick(other, p["component_id"], p.get("ref_group", ""))`. Add `ref_group` to the "Dict keys" list in the module docstring.

- [ ] **Step 6: Write the fold CSVs and fold references in prepare_data**

In `scripts/prepare_data.py`, change `from refer_or_clear.data.splits import protocol_a, protocol_b` to `from refer_or_clear.data.splits import protocol_a_kfold, protocol_b` (Task 3 and Task 4 change this line again). Replace the step-7 block

```python
    print("[7/7] splits")
    pa, pb = protocol_a(rec, raw), protocol_b(rec)
    pa.to_csv(out / "splits" / "protocol_a.csv", index=False)
    pb.to_csv(out / "splits" / "protocol_b.csv", index=False)
```

with

```python
    print("[7/7] splits and per-fold references")
    for old in (out / "splits").glob("protocol_a*.csv"):
        old.unlink()
    folds_a = protocol_a_kfold(rec, raw)
    fold_refs = []
    for k, df in folds_a.items():
        df.to_csv(out / "splits" / f"protocol_a_it{k}.csv", index=False)
        train_ids = set(df[df.role == "train"].image_id)
        fold_refs.append(build_references(rec[rec.image_id.isin(train_ids)], out / "canvases", out / "references",
                                          group=f"it{k}", write_all=False))
    fold_refs = pd.concat(fold_refs, ignore_index=True)
    pa = pd.concat(folds_a.values(), ignore_index=True)
    pb = protocol_b(rec)
    pb.to_csv(out / "splits" / "protocol_b.csv", index=False)
```

and in the `counts.md` list, after the `"\n\n## References\n", md(refs.set_index("design_id")),` entry, add
`"\n\n## Protocol A per-fold references (training files only)\n", md(fold_refs.groupby("group").agg(designs=("design_id", "size"), min_half=("n_fold0", "min"), min_half1=("n_fold1", "min"), min_coverage=("min_coverage", "min"))),`.
Change the loop `for name, df in (("Protocol A", pa), ("Protocol B", pb)):` to group Protocol A by fold:
`for name, df in (("Protocol A (all folds stacked)", pa), ("Protocol B", pb)):` and, for Protocol A, use
`df.groupby(["ref_group", "role", "label"]).size().unstack(fill_value=0)` when `"ref_group" in df` and the existing grouping otherwise:

```python
    for name, df in (("Protocol A (all folds stacked)", pa), ("Protocol B", pb)):
        keys = ["ref_group", "role", "label"] if "ref_group" in df and df.ref_group.astype(bool).any() else ["role", "subset", "label"]
        L += [f"\n\n## {name}\n", md(df.groupby(keys).size().unstack(fill_value=0))]
```

- [ ] **Step 7: Run the test**

Run: `$PY tests/test_splits_anchor.py`
Expected: `ok`. (It reads the raw fold CSVs and `data/processed/records.csv`, both present locally; it does not need rebuilt canvases.)

- [ ] **Step 8: Commit**

```bash
git add src/refer_or_clear/data/splits.py src/refer_or_clear/data/references.py src/refer_or_clear/data/pairs.py \
        scripts/prepare_data.py tests/test_splits_anchor.py
git commit -m "Protocol A: the paper's 10 template folds as given, per-fold medians from training files only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Protocol B splits (the paper's 10 nationality groups)

**Files:**
- Modify: `src/refer_or_clear/data/splits.py` (add `FSL_GROUPS`, `NATIONALITY`, `protocol_b_groups`)
- Modify: `scripts/prepare_data.py` (write `protocol_b_g{k}.csv`)
- Test: `tests/test_splits_heldout.py` (create)

**Interfaces:**
- Consumes: `COLS`, `REF_COL`, `_train_val` from `splits.py`.
- Produces: `splits.FSL_GROUPS: tuple[str, ...]` (the ten names, in the paper's order); `splits.protocol_b_groups(records: pd.DataFrame) -> dict[int, pd.DataFrame]` with `protocol == "B"` and `ref_group == ""`; files `data/processed/splits/protocol_b_g{k}.csv`, k = 0..9 (k indexes `FSL_GROUPS`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_splits_heldout.py`:

```python
"""Protocol B nationality groups: python tests/test_splits_heldout.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear.data.splits import FSL_GROUPS, NATIONALITY, REF_COL, protocol_b_groups  # noqa: E402

RECORDS = pd.read_csv(ROOT / "data" / "processed" / "records.csv", keep_default_na=False)

PAPER = ("svk_alb_aze_lva", "svk_rus_esp_aze", "grc_esp_fin_srb", "svk_alb_grc_est", "alb_grc_rus_lva",
         "alb_grc_est_rus", "svk_rus_lva_fin", "rus_aze_lva_fin", "svk_alb_lva_fin", "svk_alb_rus_fin")


def test_groups_are_the_papers():
    assert FSL_GROUPS == PAPER


def test_each_group_holds_out_its_four_nationalities():
    groups = protocol_b_groups(RECORDS)
    assert sorted(groups) == list(range(10))
    tested = set()
    for k, df in groups.items():
        want = {NATIONALITY[n] for n in FSL_GROUPS[k].split("_")}
        assert set(df[df.role == "test"].design_id) == want, k
        assert not (set(df[df.role != "test"].design_id) & want), k
        assert set(df.role) == {"train", "val", "test"}, k
        assert (df.protocol == "B").all() and (df[REF_COL] == "").all()
        tr, va = set(df[df.role == "train"].component_id), set(df[df.role == "val"].component_id)
        assert not (tr & va), "a component is split across train and val"
        frac = (df.role == "val").sum() / (df.role != "test").sum()
        assert 0.05 < frac < 0.15, frac
        tested |= want
    assert len(tested) == 10, "every nationality is tested in at least one group"


if __name__ == "__main__":
    test_groups_are_the_papers()
    test_each_group_holds_out_its_four_nationalities()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_splits_heldout.py`
Expected: `ImportError: cannot import name 'FSL_GROUPS'`

- [ ] **Step 3: Implement**

In `src/refer_or_clear/data/splits.py`, add below `protocol_a_kfold`:

```python
# The ten nationality groups hard-coded in the SIDTD paper's few-shot code
# (SIDTD/models/fsl_models/test_fsl.py, partitions_test); each names the four test nationalities.
FSL_GROUPS = ("svk_alb_aze_lva", "svk_rus_esp_aze", "grc_esp_fin_srb", "svk_alb_grc_est", "alb_grc_rus_lva",
              "alb_grc_est_rus", "svk_rus_lva_fin", "rus_aze_lva_fin", "svk_alb_lva_fin", "svk_alb_rus_fin")
NATIONALITY = {"alb": "sidtd/alb_id", "aze": "sidtd/aze_passport", "esp": "sidtd/esp_id", "est": "sidtd/est_id",
               "fin": "sidtd/fin_id", "grc": "sidtd/grc_passport", "lva": "sidtd/lva_passport",
               "rus": "sidtd/rus_internalpassport", "srb": "sidtd/srb_passport", "svk": "sidtd/svk_id"}


def protocol_b_groups(records: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Protocol B: group k -> split frame; the four nationalities of FSL_GROUPS[k] are the test set, the other six
    train, with a 10 percent validation set by provenance-component hash. References use the global fold medians."""
    s0 = records[records.dataset == "sidtd"]
    out = {}
    for k, name in enumerate(FSL_GROUPS):
        test_designs = {NATIONALITY[n] for n in name.split("_")}
        s = s0.copy()
        s["protocol"], s["subset"], s[REF_COL] = "B", "", ""
        is_test = s.design_id.isin(test_designs)
        s["role"] = "test"
        s.loc[~is_test, "role"] = _train_val(s[~is_test])
        if set(s[s.role == "test"].design_id) & set(s[s.role != "test"].design_id):
            raise ValueError(f"group {name}: a design is both test and train")
        out[k] = s[COLS + [REF_COL]].reset_index(drop=True)
    return out
```

In `scripts/prepare_data.py`, extend the splits import to `protocol_a_kfold, protocol_b, protocol_b_groups` and, in step 7 after the Protocol A loop, add:

```python
    for old in (out / "splits").glob("protocol_b_g*.csv"):
        old.unlink()
    groups_b = protocol_b_groups(rec)
    for k, df in groups_b.items():
        df.to_csv(out / "splits" / f"protocol_b_g{k}.csv", index=False)
    pbg = pd.concat(groups_b.values(), ignore_index=True)
```

and add `("Protocol B, held-out nationalities (all groups stacked)", pbg)` to the counts loop (it groups by `role, subset, label` because its `ref_group` is empty).

- [ ] **Step 4: Run the test**

Run: `$PY tests/test_splits_heldout.py && $PY tests/test_splits_anchor.py`
Expected: `ok` twice.

- [ ] **Step 5: Commit**

```bash
git add src/refer_or_clear/data/splits.py scripts/prepare_data.py tests/test_splits_heldout.py
git commit -m "Protocol B: the paper's 10 held-out nationality groups

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Configurations and protocol renumbering (FantasyID becomes C, IDNet D)

**Files:**
- Rename: `configs/protocol_c.yaml` to `configs/protocol_d.yaml`, then `configs/protocol_b.yaml` to `configs/protocol_c.yaml`
- Rewrite: `configs/protocol_a.yaml`; Create: `configs/protocol_b.yaml`
- Modify: `src/refer_or_clear/data/splits.py` (rename `protocol_b` to `protocol_c`)
- Modify: `scripts/prepare_data.py` (FantasyID split written as `protocol_c.csv`)
- Modify: `scripts/cluster/stage-data.sh` (split check)
- Test: `tests/test_configs.py` (create)

**Interfaces:**
- Produces: config keys used by Task 5: `data.split` with a `{fold}` placeholder in A and B, `data.folds: 10`, `train.select: accuracy`, `augment.jpeg_quality: null`, `train.seeds: [42]`. `splits.protocol_c(records)` (formerly `protocol_b`) with `protocol == "C"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_configs.py`:

```python
"""Protocol configs: python tests/test_configs.py"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(letter):
    return yaml.safe_load(open(ROOT / "configs" / f"protocol_{letter}.yaml"))


def test_letters_match_files():
    for letter in "abcd":
        assert load(letter)["protocol"] == letter.upper(), letter


def test_anchor_and_heldout_recipe():
    for letter, split in (("a", "protocol_a_it{fold}.csv"), ("b", "protocol_b_g{fold}.csv")):
        c = load(letter)
        assert Path(c["data"]["split"]).name == split, c["data"]["split"]
        assert c["data"]["folds"] == 10
        assert c["augment"]["jpeg_quality"] is None
        assert "canvas_jpeg_quality" not in c["augment"]
        assert c["train"]["select"] == "accuracy"
        assert c["train"]["seeds"] == [42]
        assert c["train"]["epochs"] == 40
        assert c["eval"]["test_variants"] == ["native", "equalized_qf75"]


def test_fantasyid_and_idnet_moved():
    assert Path(load("c")["data"]["split"]).name == "protocol_c.csv"
    assert "FantasyID" in open(ROOT / "configs" / "protocol_c.yaml").read().splitlines()[0]
    assert Path(load("d")["data"]["split"]).name == "protocol_d.csv"
    assert "IDNet" in open(ROOT / "configs" / "protocol_d.yaml").read().splitlines()[0]


if __name__ == "__main__":
    test_letters_match_files()
    test_anchor_and_heldout_recipe()
    test_fantasyid_and_idnet_moved()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_configs.py`
Expected: `FileNotFoundError` for `protocol_d.yaml`.

- [ ] **Step 3: Rename the FantasyID and IDNet configs**

```bash
git mv configs/protocol_c.yaml configs/protocol_d.yaml
git mv configs/protocol_b.yaml configs/protocol_c.yaml
```

In `configs/protocol_d.yaml`: first line `# Protocol C: IDNet published cross-type pair. ...` becomes `# Protocol D: IDNet published cross-type pair. ...`; `protocol: C` becomes `protocol: D`; `split: data/processed/splits/protocol_c.csv` becomes `split: data/processed/splits/protocol_d.csv`; delete the `canvas_jpeg_quality: 95 ...` line (the canvases are PNG now; the C and D recipes are revisited in their own plan).
In `configs/protocol_c.yaml`: first line becomes `# Protocol C: train on all 10 MIDV designs, test zero-shot on FantasyID's official test split`; `protocol: B` becomes `protocol: C`; `split: ...protocol_b.csv` becomes `split: data/processed/splits/protocol_c.csv`; delete the `canvas_jpeg_quality` line.

- [ ] **Step 4: Write the Protocol A and B configs**

Replace `configs/protocol_a.yaml` with:

```yaml
# Protocol A (anchor): the SIDTD paper's template 10-fold protocol (Boned et al., Scientific Data 11:1356,
# 2024, Table 5, Composite Detection, K-fold, Balanced). Raw files, the official folds as given, accuracy
# and ROC AUC over folds. Spec: docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md
protocol: A
data:
  split: data/processed/splits/protocol_a_it{fold}.csv
  folds: 10
  canvas: [512, 800]
  reference: median             # median | specimen; the control (fusion single) ignores it
  wrong_design_fraction: 0.1    # reference arms only; the single-image control gets none (train.py)
model:
  arch: changerex_r18
  fusion: difference
  init: levir
train:
  epochs: 40
  patience: null                # no early stopping: every arm trains the full schedule
  select: accuracy              # best validation accuracy, ties by validation AUC (the paper selects on accuracy)
  batch_size: 8
  lr_backbone: 1.0e-4
  lr_new: 1.0e-3
  weight_decay: 0.05
  warmup_iters: 1000
  mask_loss_weight: 3.0
  amp: true
  seeds: [42]
augment:
  crop: null                    # full 512 x 800 canvas
  affine_px: 8
  jpeg_quality: null            # raw files as the paper reads them: no JPEG augmentation, no re-encoding
  photometric: true
eval:
  test_variants: [native, equalized_qf75]   # equalized: query and reference canvases re-encoded at 75
  score_orientation: bonafide_high          # 1 = genuine, threshold 0.5 with >= (DeepID)
  threshold_source: source_validation
  metrics: [accuracy, auc, eer, pixel_f1]
```

Create `configs/protocol_b.yaml` with the same content except:
- the three header lines become
  `# Protocol B: held-out nationalities. The SIDTD paper's ten nationality groups (SIDTD/models/fsl_models/test_fsl.py),`
  `# four test nationalities per group, the other six train; same recipe as Protocol A. Mean and sd over groups.`
  `# Spec: docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md`
- `protocol: B`
- `split: data/processed/splits/protocol_b_g{fold}.csv`

- [ ] **Step 5: Rename the FantasyID split builder**

In `src/refer_or_clear/data/splits.py`, rename `def protocol_b(records` to `def protocol_c(records`, change its three `"B"` literals (`pool["protocol"], pool["subset"] = "B", ""` and `f["protocol"], f["role"], f["subset"] = "B", "", ""`) to `"C"`, and change "Protocol B:" in the docstring paragraph about FantasyID to "Protocol C:". In `scripts/prepare_data.py`, change the import to `protocol_a_kfold, protocol_b_groups, protocol_c`, replace `pb = protocol_b(rec)` and `pb.to_csv(out / "splits" / "protocol_b.csv", index=False)` with `pc = protocol_c(rec)` and `pc.to_csv(out / "splits" / "protocol_c.csv", index=False)`, add `(out / "splits" / "protocol_b.csv").unlink(missing_ok=True)`, and change the counts entry `("Protocol B", pb)` to `("Protocol C (FantasyID)", pc)`. In the module docstring, replace `splits/protocol_a.csv, splits/protocol_b.csv` with `splits/protocol_a_it{0..9}.csv, splits/protocol_b_g{0..9}.csv, splits/protocol_c.csv`.

- [ ] **Step 6: Update the staging check**

In `scripts/cluster/stage-data.sh`, replace the Python list
`["data/processed/splits/protocol_a.csv", "data/processed/splits/protocol_b.csv", "data/processed/splits/protocol_c.csv"]` with
`[f"data/processed/splits/protocol_a_it{k}.csv" for k in range(10)] + [f"data/processed/splits/protocol_b_g{k}.csv" for k in range(10)] + ["data/processed/splits/protocol_c.csv"]`
and delete the `# TODO: replace with the real counts ...` comment line above it.

- [ ] **Step 7: Run the tests**

Run: `$PY tests/test_configs.py && $PY tests/test_splits_anchor.py && $PY tests/test_splits_heldout.py`
Expected: `ok` three times.

- [ ] **Step 8: Commit**

```bash
git add -A configs src/refer_or_clear/data/splits.py scripts/prepare_data.py scripts/cluster/stage-data.sh tests/test_configs.py
git commit -m "Configs: A anchor, B held-out nationalities; FantasyID becomes C, IDNet D

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Training and evaluation: folds, the paper's checkpoint rule, no JPEG augmentation, control without wrong-design pairs

**Files:**
- Modify: `src/refer_or_clear/train.py`
- Modify: `src/refer_or_clear/evaluate.py` (`load_split`, `role_pairs`, CLI)
- Test: `tests/test_train_config.py` (create)

**Interfaces:**
- Consumes: config keys from Task 4.
- Produces: `train.py --fold K`; `train.SINGLE_FUSIONS = ("single", "attention_single")`; `train.selection_key(select: str, val_acc, val_auc, val_loss, train_loss) -> tuple`; `train.jpeg_range(rc: dict) -> tuple[int, int] | None`; `resolve()` returns keys `fold`, `select`, `wrong_design_fraction` (0.0 for single fusions); `evaluate.load_split(cfg, data_root, config_path=None, fold=None)`; `evaluate.role_pairs(..., limit=None, wrong_design_fraction=None)`; `metrics.json` gains `"fold"`; `log.csv` gains `val_acc`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_config.py`:

```python
"""train.resolve, the selection rule and fold splits: python tests/test_train_config.py"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear import evaluate as E  # noqa: E402
from refer_or_clear import train as T  # noqa: E402

CFG_A = E.load_config(ROOT / "configs" / "protocol_a.yaml")


def rc_for(fusion, fold=3):
    a = T.parse_args(["--config", "configs/protocol_a.yaml", "--arch", "changerex_r18", "--fusion", fusion,
                      "--init", "levir", "--run-dir", "/tmp/x", "--fold", str(fold)])
    return T.resolve(a, CFG_A)


def test_control_gets_no_wrong_design_pairs():
    assert rc_for("single")["wrong_design_fraction"] == 0.0
    assert rc_for("attention_single")["wrong_design_fraction"] == 0.0
    assert rc_for("difference")["wrong_design_fraction"] == 0.1


def test_fold_select_and_jpeg():
    rc = rc_for("difference", fold=7)
    assert rc["fold"] == 7 and rc["select"] == "accuracy"
    assert T.jpeg_range(rc) is None
    assert T.jpeg_range({"augment": {"jpeg_quality": [60, 100]}}) == (60, 100)


def test_selection_key():
    k = T.selection_key
    assert k("accuracy", 0.99, 0.90, 0.1, 0.1) > k("accuracy", 0.98, 1.00, 0.1, 0.1)   # accuracy first
    assert k("accuracy", 0.99, 0.95, 0.1, 0.1) > k("accuracy", 0.99, 0.94, 0.1, 0.1)   # AUC breaks ties
    assert not (k("accuracy", 1.0, 1.0, 0.1, 0.1) > k("accuracy", 1.0, 1.0, 0.1, 0.1))  # full tie keeps the earlier
    assert k("auc", 0.5, 0.9, 0.1, 0.1) > k("auc", 0.9, 0.8, 0.1, 0.1)
    assert k("accuracy", None, None, 0.2, 0.1) > k("accuracy", None, None, 0.3, 0.1)
    assert k("accuracy", 0.9, None, 0.2, 0.1) == (0.9, -math.inf)


def test_load_split_formats_the_fold():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "splits").mkdir()
        pd.DataFrame([dict(image_id="i", protocol="A", role="test", subset="", component_id="c",
                           design_id="x/d", label=0, ref_group="it3")]).to_csv(d / "splits" / "protocol_a_it3.csv", index=False)
        pd.DataFrame([dict(image_id="i")]).to_csv(d / "records.csv", index=False)
        split, _ = E.load_split(CFG_A, d, fold=3)
        assert list(split.ref_group) == ["it3"]
        try:
            E.load_split(CFG_A, d)
        except ValueError as e:
            assert "fold" in str(e)
        else:
            raise AssertionError("a {fold} split without --fold was accepted")


def test_role_pairs_override_reaches_build_pairs():
    seen = {}

    def fake(split_df, records_df, root, reference, wrong_design_fraction, role, seed):
        seen["frac"] = wrong_design_fraction
        return []
    import refer_or_clear.data.pairs as P
    orig, P.build_pairs = P.build_pairs, fake
    try:
        split = pd.DataFrame([dict(role="train")])
        E.role_pairs(split, None, "x", "train", "median", CFG_A, 42, wrong_design_fraction=0.0)
        assert seen["frac"] == 0.0
        E.role_pairs(split, None, "x", "train", "median", CFG_A, 42)
        assert seen["frac"] == 0.1
    finally:
        P.build_pairs = orig


if __name__ == "__main__":
    test_control_gets_no_wrong_design_pairs()
    test_fold_select_and_jpeg()
    test_selection_key()
    test_load_split_formats_the_fold()
    test_role_pairs_override_reaches_build_pairs()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_train_config.py`
Expected: `error: unrecognized arguments: --fold 3` (SystemExit from argparse).

- [ ] **Step 3: Implement `evaluate.py` changes**

Replace `load_split` with:

```python
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
```

Replace `role_pairs` with:

```python
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
```

In `main()` of `evaluate.py`: add `ap.add_argument("--fold", type=int, default=None, help="fold or group index for Protocols A and B (default: checkpoint config)")`; after loading the checkpoint, set `fold = args.fold if args.fold is not None else ck.get("config", {}).get("fold")`; change `split_df, records_df = load_split(cfg, args.data_root)` to `split_df, records_df = load_split(cfg, args.data_root, fold=fold)`; and add `"fold": fold,` to the `out` dict. Validation and test roles already get fraction 0.

- [ ] **Step 4: Implement `train.py` changes**

1. Add `"val_acc"` to `LOG_COLUMNS` directly after `"val_loss"`, and below `GRAD_CLIP = 5.0` add:

```python
SINGLE_FUSIONS = ("single", "attention_single")   # ignore the reference: no wrong-design pairs for them


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
```

2. In `parse_args`, add `ap.add_argument("--fold", type=int, default=None, help="fold (Protocol A) or group (Protocol B) index")`.
3. In `resolve`, compute `fusion = args.fusion or m.get("fusion", "difference")` before the `return`, use `fusion=fusion` in the dict, and add these entries:

```python
        fold=args.fold,
        select=str(t.get("select", "auc")),
        wrong_design_fraction=0.0 if fusion in SINGLE_FUSIONS
        else float(cfg.get("data", {}).get("wrong_design_fraction", 0.1)),
```

   replacing the existing `wrong_design_fraction=...` line.
4. In `main`, change `split_df, records_df = E.load_split(cfg, args.data_root)` to `split_df, records_df = E.load_split(cfg, args.data_root, fold=rc["fold"])`; in the `pairs = {...}` comprehension pass `wrong_design_fraction=rc["wrong_design_fraction"]` to `E.role_pairs`; in `make_loader(...)` for training replace `jpeg=tuple(aug.get("jpeg_quality", (60, 100)))` with `jpeg=jpeg_range(rc)`.
5. In the epoch loop: replace `val_loss = val_auc = None` with `val_loss = val_auc = val_acc = None`; after `val_loss, val_auc = vp["loss"], M.auc(vp["score"], vp["label"])` add `val_acc = M.deepid_acc(vp["score"], vp["label"])`; add `val_acc=val_acc` to `log.log(...)`; add `val_acc {'na' if val_acc is None else f'{val_acc:.4f}'}` to the printed line after `val_auc`. Replace

```python
        score = val_auc if val_auc is not None else (-val_loss if val_loss is not None else -tr["loss"])
```

   with

```python
        score = selection_key(rc["select"], val_acc, val_auc, val_loss, tr["loss"])
```

   and initialise `best_score, best_epoch, bad_epochs = None, 0, 0`; change `if score > best_score:` to `if best_score is None or score > best_score:`.
6. Add `"fold": rc["fold"],` to the `metrics` dict after `"seed": rc["seed"],`. Update the module docstring: "the checkpoint with the best validation AUC is evaluated" becomes "the checkpoint is chosen by train.select: best validation accuracy with AUC breaking ties (Protocols A and B, the SIDTD paper's rule) or best validation AUC".

- [ ] **Step 5: Run the tests**

Run: `$PY tests/test_train_config.py && $PY tests/test_metrics.py`
Expected: `ok`, then the existing metrics output.

- [ ] **Step 6: Commit**

```bash
git add src/refer_or_clear/train.py src/refer_or_clear/evaluate.py tests/test_train_config.py
git commit -m "Training: --fold, best validation accuracy (AUC breaks ties), JPEG augmentation off, control without wrong-design pairs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Sweep generator: folds and protocol D

**Files:**
- Modify: `scripts/make_sweep.py`
- Test: `tests/test_make_sweep.py` (create)

**Interfaces:**
- Consumes: `train.py --fold` (Task 5); configs (Task 4).
- Produces: `--folds` option (`"0-9"` or `"0,3,5"`; empty for none); tags `<P>-<arch>-<fusion>-<reference>-<init>-f<k>-s<seed>` when a fold is given, unchanged otherwise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_make_sweep.py`:

```python
"""Sweep generator: python tests/test_make_sweep.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_sweep as MS  # noqa: E402


def run(*extra):
    d = tempfile.mkdtemp()
    MS.write_sweep(["--archs", "changerex_r18", "--references", "median", "--inits", "levir", "--seeds", "42",
                    "--name", "t", "--out", d, *extra])
    return {p.name: p.read_text() for p in Path(d).glob("*.sh") if p.name != "submit_all.sh"}


def test_folds_in_tags_and_flags():
    jobs = run("--protocols", "A", "--fusions", "single,difference", "--folds", "0-1")
    assert set(jobs) == {f"A-cex-{f}-median-levir-f{k}-s42.sh" for f in ("single", "difference") for k in (0, 1)}
    text = jobs["A-cex-single-median-levir-f1-s42.sh"]
    assert "--fold 1" in text and "configs/protocol_a.yaml" in text
    assert "--seed 42 \\\n        --fold 1 \\\n        --data-root" in text, "line continuation broken"
    assert 'RUN_DIR="results/runs/A-cex-single-median-levir-f1-s42"' in text
    assert "protocol_a.csv" not in text


def test_no_folds_keeps_old_tags():
    jobs = run("--protocols", "C", "--fusions", "difference")
    assert set(jobs) == {"C-cex-difference-median-levir-s42.sh"}
    assert "--fold" not in jobs["C-cex-difference-median-levir-s42.sh"]
    assert "configs/protocol_c.yaml" in jobs["C-cex-difference-median-levir-s42.sh"]


def test_protocol_d_and_fold_list():
    jobs = run("--protocols", "D", "--fusions", "difference", "--folds", "2,5")
    assert set(jobs) == {"D-cex-difference-median-levir-f2-s42.sh", "D-cex-difference-median-levir-f5-s42.sh"}
    assert "configs/protocol_d.yaml" in jobs["D-cex-difference-median-levir-f2-s42.sh"]


if __name__ == "__main__":
    test_folds_in_tags_and_flags()
    test_no_folds_keeps_old_tags()
    test_protocol_d_and_fold_list()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_make_sweep.py`
Expected: `error: unrecognized arguments: --folds 0-1`

- [ ] **Step 3: Implement**

In `scripts/make_sweep.py`:
1. `CHOICES["protocols"] = ("A", "B", "C", "D")`; `CONFIG = {"A": "configs/protocol_a.yaml", "B": "configs/protocol_b.yaml", "C": "configs/protocol_c.yaml", "D": "configs/protocol_d.yaml"}`.
2. Add `fold: int | None = None` as the last field of `Arm` and change `tag` to:

```python
    @property
    def tag(self) -> str:
        f = f"-f{self.fold}" if self.fold is not None else ""
        return f"{self.protocol}-{SHORT_ARCH[self.arch]}-{self.fusion}-{self.reference}-{self.init}{f}-s{self.seed}"
```

3. Replace `expand_arms` with:

```python
def expand_arms(protocols, archs, fusions, references, inits, seeds, folds=(None,)) -> list[Arm]:
    """The grid with seeds innermost, then folds, so a job cut short leaves whole cells."""
    for name, values in dict(protocols=protocols, archs=archs, fusions=fusions,
                             references=references, inits=inits).items():
        bad = [v for v in values if v not in CHOICES[name]]
        if bad:
            raise ValueError(f"unknown {name} {bad}; expected one of {CHOICES[name]}")
    return [Arm(p, b, f, r, t, int(s), k)
            for p, b, f, r, t, k, s in itertools.product(protocols, archs, fusions, references, inits, folds, seeds)]


def parse_folds(s: str) -> list:
    """'' -> [None]; '0-9' -> [0..9]; '2,5' -> [2, 5]."""
    s = s.strip()
    if not s:
        return [None]
    if "-" in s:
        lo, hi = (int(x) for x in s.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in _csv(s)]
```

4. In `ARM`, add a `{fold_flag}` placeholder directly after `--seed {seed} \\` so the block reads:

```
        --seed {seed} \\{fold_flag}
        --data-root "$ROC_DATA" \\
```

   and in `render_job` pass `fold_flag="" if a.fold is None else f"\n        --fold {a.fold} \\"` (one backslash character in the rendered text: `ARM` is not a raw string, so its own `\\` already renders as one, and `str.format` inserts `fold_flag` verbatim). The rendered script must contain `        --fold 1 \` followed directly by the `--data-root` line; a double backslash there would end the command early, which `bash -n` does not detect.
5. In `PREAMBLE`, replace
   `[[ -f data/processed/splits/protocol_a.csv ]] || {{`
   with
   `[[ -n "$(ls data/processed/splits/*.csv 2>/dev/null)" ]] || {{`.
6. In `parse_args`, add `p.add_argument("--folds", default="", help="fold or group indices, e.g. 0-9 or 2,5 (Protocols A and B); empty for none")`; in `write_sweep`, pass `parse_folds(a.folds)` as the new last argument of `expand_arms`.
7. Update the usage examples in the module docstring to:

```
  # Protocol A (anchor): control and the two reference arms on the paper's 10 folds, one GPU per arm
  python scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions single --references median \
      --inits levir --seeds 42 --folds 0-9 --time 6:00:00 --name A-anchor-control
  python scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions difference --references median,specimen \
      --inits levir --seeds 42 --folds 0-9 --time 6:00:00 --name A-anchor-ref
```

- [ ] **Step 4: Run the test**

Run: `$PY tests/test_make_sweep.py`
Expected: `ok`. Then `bash -n` every generated script of one throwaway sweep to check shell syntax:
`$PY scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions single --references median --inits levir --seeds 42 --folds 0-1 --name syntax --out /tmp/syntax && for f in /tmp/syntax/*.sh; do bash -n "$f" || echo "BAD $f"; done`
Expected: no `BAD` line.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_sweep.py tests/test_make_sweep.py
git commit -m "Sweeps: --folds and protocol D; fold in run tags

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Aggregator: folds, the Table 5 layout, per-nationality results and the gate

**Files:**
- Modify: `scripts/aggregate_runs.py`
- Test: `tests/test_aggregate.py` (create)

**Interfaces:**
- Consumes: `metrics.json` with `fold` and `test.{native,equalized_qf75}.{auc,deepid_acc}` (Task 5); tags from Task 6.
- Produces: `aggregate_runs.load(runs, protocol) -> DataFrame` (column `fold`; runs without a fold are skipped for A and B); `paper_table(df, variant) -> str`; `per_design(df) -> str`; `gate(df) -> tuple[bool, str]`; CLI `--gate` (exit code 1 on FAIL).

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate.py`:

```python
"""Aggregator: python tests/test_aggregate.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_runs as AG  # noqa: E402


def write(runs: Path, tag: str, auc: float, acc: float, fold, designs=()):
    d = runs / tag
    d.mkdir(parents=True)
    block = {"auc": auc, "deepid_acc": acc}
    subsets = {f"design:{x}": {"native": {"auc": auc - 0.01, "deepid_acc": acc}} for x in designs}
    json.dump({"fold": fold, "val": {"auc": 0.99}, "timing": {"epochs": 40, "best_epoch": 30},
               "test": {"native": block, "equalized_qf75": block}, "subsets": subsets}, open(d / "metrics.json", "w"))


def make(control_auc=0.995, n=10):
    runs = Path(tempfile.mkdtemp())
    for k in range(n):
        write(runs, f"A-cex-single-median-levir-f{k}-s42", control_auc, 0.985, k)
        write(runs, f"A-cex-difference-median-levir-f{k}-s42", 0.999, 0.995, k)
    write(runs, "A-cex-single-median-levir-s42", 0.40, 0.40, None)   # superseded, no fold
    return runs


def test_superseded_runs_are_ignored():
    df = AG.load(make(), "A")
    assert set(df.fold.dropna().astype(int)) == set(range(10))
    assert not ((df.fusion == "single") & (df.auc < 0.5)).any()


def test_paper_table_has_published_rows_and_ours():
    t = AG.paper_table(AG.load(make(), "A"), "native")
    for name in ("EfficientNet-B3", "ResNet50", "ViT-L/16", "TransFG", "CoAARN", "control", "design median"):
        assert name in t, name
    assert "99.4 ± 1.0" in t and "98.5 ± 0.0" in t


def test_gate():
    ok, msg = AG.gate(AG.load(make(), "A"))
    assert ok, msg
    ok, msg = AG.gate(AG.load(make(control_auc=0.95), "A"))
    assert not ok and "AUC" in msg
    ok, msg = AG.gate(AG.load(make(n=9), "A"))
    assert not ok and "9 of 10" in msg


def test_per_design_for_heldout():
    runs = Path(tempfile.mkdtemp())
    write(runs, "B-cex-difference-median-levir-f0-s42", 0.9, 0.9, 0, designs=("sidtd/alb_id", "sidtd/aze_passport"))
    write(runs, "B-cex-difference-median-levir-f1-s42", 0.8, 0.8, 1, designs=("sidtd/alb_id",))
    t = AG.per_design(AG.load(runs, "B"))
    assert "sidtd/alb_id" in t and "0.840" in t and "sidtd/aze_passport" in t


if __name__ == "__main__":
    test_superseded_runs_are_ignored()
    test_paper_table_has_published_rows_and_ours()
    test_gate()
    test_per_design_for_heldout()
    print("ok")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `$PY tests/test_aggregate.py`
Expected: `AttributeError: module 'aggregate_runs' has no attribute 'paper_table'`

- [ ] **Step 3: Implement**

In `scripts/aggregate_runs.py`:
1. Replace `TAG` with

```python
TAG = re.compile(r"^(?P<protocol>[A-D])-(?P<arch>[a-z0-9]+)-(?P<fusion>[a-z_]+)-(?P<reference>[a-z]+)-(?P<init>[a-z]+)"
                 r"(?:-f(?P<fold>\d+))?-s(?P<seed>\d+)$")
FOLDED = {"A", "B"}   # protocols whose runs carry a fold or group index; foldless runs there are superseded
```

   and add after `ARM_LABEL`:

```python
OURS = {("single", "median"): "Ours, control (no reference)", ("difference", "median"): "Ours, design median",
        ("difference", "specimen"): "Ours, blank specimen"}
# Boned et al., Scientific Data 11:1356 (2024), Table 5, Composite Detection (K-fold, Balanced):
# (model, accuracy %, sd, ROC AUC, sd), as published.
PUBLISHED_A = [("EfficientNet-B3", 99.4, 1.0, 1.00, 0.00), ("ResNet50", 98.1, 1.2, 0.99, 0.01),
               ("ViT-L/16", 55.2, 2.3, 0.50, 0.04), ("TransFG", 96.6, 1.5, 0.99, 0.01),
               ("CoAARN", 98.6, 1.6, 0.99, 0.01)]
GATE = dict(auc=0.99, acc=0.97, n=10)
```

2. In `load`, skip foldless runs of folded protocols and record the fold: after `if not m or m["protocol"] != protocol: continue` add

```python
        if protocol in FOLDED and m["fold"] is None:
            continue
```

   (the `**m.groupdict()` already carries `fold`). Also add `"auc"` and `"deepid_acc"` are in `METRICS` (they are).
3. Add these functions above `main`:

```python
def _ms(v: pd.Series, scale: float = 1.0, nd: int = 3) -> str:
    v = pd.to_numeric(v, errors="coerce").dropna() * scale
    if v.empty:
        return ""
    return f"{v.mean():.{nd}f} ± {v.std(ddof=0):.{nd}f}"


def paper_table(df: pd.DataFrame, variant: str) -> str:
    """Table 5 layout: the published rows (Protocol A only), then our arms; accuracy in percent."""
    out = ["| Model | n | Accuracy (%) | ROC AUC |", "|---|---|---|---|"]
    if df.protocol.iloc[0] == "A" and variant == "native":
        out += [f"| {m} (Boned et al. 2024) | 10 | {a:.1f} ± {sa:.1f} | {u:.2f} ± {su:.2f} |" for m, a, sa, u, su in PUBLISHED_A]
    sub = df[df.variant == variant]
    for (f, r), label in OURS.items():
        s = sub[(sub.fusion == f) & (sub.reference == r)]
        if len(s):
            out.append(f"| {label} | {len(s)} | {_ms(s.deepid_acc, 100, 1)} | {_ms(s.auc)} |")
    return "\n".join(out)


def per_design(df: pd.DataFrame) -> str:
    """Per test nationality: mean over the groups in which it was tested (native variant)."""
    sub = df[df.variant.str.startswith("native:design:")].copy()
    if sub.empty:
        return "(no per-design subsets)"
    sub["design"] = sub.variant.str.replace("native:design:", "", regex=False)
    out = ["| Arm | Design | groups | Accuracy (%) | ROC AUC |", "|---|---|---|---|---|"]
    for (f, r), label in OURS.items():
        s = sub[(sub.fusion == f) & (sub.reference == r)]
        for d, g in s.groupby("design"):
            out.append(f"| {label} | {d} | {len(g)} | {_ms(g.deepid_acc, 100, 1)} | {_ms(g.auc)} |")
    return "\n".join(out)


def gate(df: pd.DataFrame) -> tuple[bool, str]:
    """The control on the raw test files: all folds present, mean AUC >= 0.99 and mean accuracy >= 0.97."""
    s = df[(df.variant == "native") & (df.fusion == "single") & (df.reference == "median")]
    n, auc_m, acc_m = len(s), pd.to_numeric(s.auc).mean(), pd.to_numeric(s.deepid_acc).mean()
    if n < GATE["n"]:
        return False, f"GATE FAIL: control has {n} of {GATE['n']} folds"
    fails = []
    if not auc_m >= GATE["auc"]:
        fails.append(f"AUC {auc_m:.4f} < {GATE['auc']}")
    if not acc_m >= GATE["acc"]:
        fails.append(f"accuracy {acc_m:.4f} < {GATE['acc']}")
    if fails:
        return False, "GATE FAIL: " + "; ".join(fails)
    return True, f"GATE PASS: control AUC {auc_m:.4f}, accuracy {acc_m:.4f} over {n} folds"
```

4. In `main`: add `ap.add_argument("--gate", action="store_true", help="print the Protocol A gate and exit 1 on FAIL")`. Change the title line to count folds: `parts = [f"# Protocol {a.protocol}: {df[df.variant == 'native'].shape[0]} runs\n"]`; insert, before the detailed per-variant tables, `parts += ["\n## Table 5 layout, raw test files\n", paper_table(df, "native"), "\n\n## Table 5 layout, compression-equalized test\n", paper_table(df, "equalized_qf75")]`; for protocol B add `parts += ["\n\n## Per test nationality\n", per_design(df)]`; add `"fold"` to the per-run columns (`df[df.variant == "native"][["fusion", "reference", "fold", "seed", ...]]`). At the end, after printing:

```python
    if a.gate:
        ok, msg = gate(df)
        print(msg)
        raise SystemExit(0 if ok else 1)
```

- [ ] **Step 4: Run the test**

Run: `$PY tests/test_aggregate.py`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/aggregate_runs.py tests/test_aggregate.py
git commit -m "Aggregator: folds, Table 5 layout with the published rows, per-nationality table, gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Rebuild the processed data, retire the superseded runs, sync to the cluster

**Files:**
- Move: `results/runs/A-cex-*` and `results/tables/protocol_a.md` to `results/superseded/2026-10-10-heldout-reencoded/`
- Modify: `tests/test_data_pairs.py` (new split names, PNG, group references)
- Modify: `docs/results-protocol-a.md` (superseded banner)
- Regenerate (gitignored): `data/processed/`

**Interfaces:**
- Consumes: everything from Tasks 1 to 4.
- Produces: `data/processed` with PNG canvases, `splits/protocol_a_it{0..9}.csv`, `splits/protocol_b_g{0..9}.csv`, `splits/protocol_c.csv`, 200 per-fold reference files (`*_it{k}_fold{0,1}.png`), synced to `/shared/b00090279/refer-or-clear/data/processed/`.

- [ ] **Step 1: Retire the superseded runs**

```bash
mkdir -p results/superseded/2026-10-10-heldout-reencoded
git mv results/runs/A-cex-* results/superseded/2026-10-10-heldout-reencoded/
git mv results/tables/protocol_a.md results/superseded/2026-10-10-heldout-reencoded/protocol_a.md
```

Add this as the first lines of `docs/results-protocol-a.md`:

```
> **Superseded (10 Oct 2026).** These runs held out the official few-shot nationalities with re-encoded images and
> JPEG augmentation, and the control received wrong-design pairs. Protocol A is now the SIDTD paper's template
> 10-fold anchor (docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md). The runs are kept in
> results/superseded/2026-10-10-heldout-reencoded/.
```

- [ ] **Step 2: Rebuild the processed data**

Run: `$PY scripts/prepare_data.py --workers 4`
Expected: the stale-JPEG message from Task 1, the seven steps, and `counts.md` printed. Then check:

```bash
ls data/processed/canvases | grep -c '\.png$'          # 5506
ls data/processed/canvases | grep -c '\.jpg$'          # 0
ls data/processed/references | grep -c '_it[0-9]_fold[01]\.png$'   # 200
ls data/processed/splits                                 # protocol_a_it0..9, protocol_b_g0..9, protocol_c.csv
du -sh data/processed
```

Expected counts as in the comments; if `canvases` holds a different number than 5,506, compare with `records.csv` rows before going on.

- [ ] **Step 3: Update the real-data pair checks**

In `tests/test_data_pairs.py`: change `split = pd.read_csv(P / "splits" / "protocol_a.csv")` to `split = pd.read_csv(P / "splits" / "protocol_a_it0.csv", keep_default_na=False)`; change both regexes `r"_fold(\d)\.jpg$"` to `r"_fold(\d)\.png$"`; and inside the `if reference == "median":` branch of the per-pair loop add `assert "_it0_fold" in p["ref_path"], p["ref_path"]`. Then in the median "no component of the sample's fold" block, restrict `comps` to the fold's training genuines: replace `comps = records[records.label == 0].groupby("design_id").component_id.apply(set)` with

```python
        train_ids = set(split[split.role == "train"].image_id)
        g = records[(records.label == 0) & records.image_id.isin(train_ids)]
        comps = g.groupby("design_id").component_id.apply(set)
```

Run: `$PY tests/test_data_pairs.py`
Expected: its pass output for train, val and test with `median` and `specimen`.

- [ ] **Step 4: Commit**

```bash
git add -A results/superseded results/runs results/tables docs/results-protocol-a.md tests/test_data_pairs.py
git commit -m "Retire the superseded Protocol A runs; pair checks on the anchor fold 0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Sync code and data to the cluster**

```bash
ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'cd /shared/b00090279/refer-or-clear && mkdir -p ../roc-superseded \
  && (mv results/runs/A-cex-* ../roc-superseded/ 2>/dev/null; true) && git pull --ff-only && git log -1 --format=%h'
rsync -a --delete --exclude '._*' --exclude '.DS_Store' data/processed/ \
  b00090279@hpc-gpu.aus.edu:/shared/b00090279/refer-or-clear/data/processed/
ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'cd /shared/b00090279/refer-or-clear && ls data/processed/splits && \
  ls data/processed/canvases | grep -c png && ls data/processed/references | grep -c "_it[0-9]_fold"'
```

Expected: the cluster HEAD equals the local HEAD; the split list as in Step 2; 5506; 200.

---

### Task 9: Cluster smoke test of the three arms

**Files:**
- Generated (not committed): `sweeps/A-smoke-control/`, `sweeps/A-smoke-ref/`

**Interfaces:**
- Consumes: Tasks 5 to 8 on the cluster.

- [ ] **Step 1: Generate and submit the smoke jobs (on the cluster login node)**

```bash
ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'export PATH=$PATH:/opt/slurm/bin; cd /shared/b00090279/refer-or-clear && \
  source /opt/miniconda/etc/profile.d/conda.sh && conda activate refer-or-clear && \
  python scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions single --references median --inits levir \
    --seeds 42 --folds 0 --suffix -smoke --extra "--epochs 1 --limit 64" --time 1:00:00 --name A-smoke-control && \
  python scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions difference --references median,specimen \
    --inits levir --seeds 42 --folds 0 --suffix -smoke --extra "--epochs 1 --limit 64" --time 1:00:00 --name A-smoke-ref && \
  bash sweeps/A-smoke-control/submit_all.sh && bash sweeps/A-smoke-ref/submit_all.sh'
```

Expected: three job ids.

- [ ] **Step 2: Wait for them and check the outputs**

Poll every few minutes until `squeue -u b00090279` no longer lists them, then:

```bash
ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'cd /shared/b00090279/refer-or-clear && \
  for t in A-cex-single-median-levir-f0-s42-smoke A-cex-difference-median-levir-f0-s42-smoke A-cex-difference-specimen-levir-f0-s42-smoke; do \
    python - "$t" <<"PY"
import json, sys
t = sys.argv[1]
m = json.load(open(f"results/runs/{t}/metrics.json")); c = json.load(open(f"results/runs/{t}/train.json"))["config"]
print(t, "fold", m["fold"], "wrong_design", c["wrong_design_fraction"], "select", c["select"],
      "jpeg", c["augment"].get("jpeg_quality"), "variants", sorted(m["test"]))
PY
  done; grep -h "val_acc" logs/slurm/A-smoke-*.out | head -3'
```

Expected: each line shows `fold 0`, `select accuracy`, `jpeg None`, `variants ['equalized_qf75', 'native']`; `wrong_design 0.0` for the control and `0.1` for the other two; the epoch log line contains `val_acc`. Any `[FAIL]` in `logs/slurm/A-smoke-*.out` stops the plan here.

- [ ] **Step 3: Remove the smoke outputs**

```bash
ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'cd /shared/b00090279/refer-or-clear && rm -rf results/runs/*-smoke checkpoints/*-smoke sweeps/A-smoke-*'
```

---

### Task 10: Protocol A campaign and the gate

**Files:**
- Generated (not committed): `sweeps/A-anchor-control/`, `sweeps/A-anchor-ref/`
- Create: `results/tables/protocol_A.md` (aggregator output), committed with `results/runs/A-*-f*-s42/{metrics.json,log.csv,train.json}`

- [ ] **Step 1: Generate and submit the 30 jobs**

```bash
ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'export PATH=$PATH:/opt/slurm/bin; cd /shared/b00090279/refer-or-clear && \
  source /opt/miniconda/etc/profile.d/conda.sh && conda activate refer-or-clear && \
  python scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions single --references median --inits levir \
    --seeds 42 --folds 0-9 --time 6:00:00 --name A-anchor-control && \
  python scripts/make_sweep.py --protocols A --archs changerex_r18 --fusions difference --references median,specimen \
    --inits levir --seeds 42 --folds 0-9 --time 6:00:00 --name A-anchor-ref && \
  bash sweeps/A-anchor-control/submit_all.sh && bash sweeps/A-anchor-ref/submit_all.sh'
```

Expected: 10 + 20 job ids.

- [ ] **Step 2: Wait for completion**

Watch in the background until 30 `results/runs/A-cex-*-f*-s42/metrics.json` exist or a log reports `[FAIL]`:

```bash
until ssh -o BatchMode=yes b00090279@hpc-gpu.aus.edu 'cd /shared/b00090279/refer-or-clear; \
  n=$(ls results/runs/A-cex-*-f*-s42/metrics.json 2>/dev/null | wc -l); \
  f=$(grep -l "^\[FAIL\]" logs/slurm/A-anchor-*.out 2>/dev/null | wc -l); [ $((n+f)) -ge 30 ]'; do sleep 300; done
```

A failed arm is resubmitted once with its own script after reading its log; the skip rule protects finished arms.

- [ ] **Step 3: Pull the results and aggregate**

```bash
rsync -az --include='A-cex-*-f*-s42/' --include='A-cex-*-f*-s42/metrics.json' --include='A-cex-*-f*-s42/log.csv' \
  --include='A-cex-*-f*-s42/train.json' --exclude='*' \
  b00090279@hpc-gpu.aus.edu:/shared/b00090279/refer-or-clear/results/runs/ results/runs/
$PY scripts/aggregate_runs.py --protocol A --gate
```

Expected: the Table 5 layout with the five published rows and our three arms, the equalized table, and a final `GATE PASS` or `GATE FAIL` line (exit code 1 on FAIL).

- [ ] **Step 4: Commit and report**

```bash
git add results/runs/A-cex-*-f*-s42 results/tables/protocol_A.md
git commit -m "Protocol A anchor: 30 runs, Table 5 layout and gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

Report the two Table 5 tables and the gate line to the user and STOP. On `GATE FAIL`, the spec's diagnostic (Section 6, item 2) is the next step, after the user agrees. Protocol B waits for the user.

---

### Task 11: Protocol B campaign (only after the gate passes and the user agrees)

**Files:**
- Generated (not committed): `sweeps/B-heldout-control/`, `sweeps/B-heldout-ref/`
- Create: `results/tables/protocol_B.md`, committed with `results/runs/B-*-f*-s42/{metrics.json,log.csv,train.json}`

- [ ] **Step 1: Generate and submit the 30 jobs**

Same commands as Task 10 Step 1 with `--protocols B`, `--name B-heldout-control` and `--name B-heldout-ref`.

- [ ] **Step 2: Wait, pull, aggregate**

Same as Task 10 Steps 2 and 3 with `B-` in place of `A-` in every glob and log name, and `$PY scripts/aggregate_runs.py --protocol B` (no `--gate`). Expected: our three arms in the Table 5 layout (no published rows) and the per-nationality table, each nationality averaged over the groups that test it.

- [ ] **Step 3: Commit and report**

```bash
git add results/runs/B-cex-*-f*-s42 results/tables/protocol_B.md
git commit -m "Protocol B held-out nationalities: 30 runs, per-nationality table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 12: Conventions and plan documents

**Files:**
- Modify: `docs/conventions.md` (JPEG row and Section 5)
- Modify: `docs/paper-plan.md` (Section 3 protocol table), mirrored to `/Users/hamza/Documents/GitHub/Thesis/00-Vault/04-Writing/2026-10-10-refer-or-clear-paper-plan.md`

- [ ] **Step 1: Conventions**

In `docs/conventions.md`, replace the table row that begins `| JPEG equalization |` with:

```
| JPEG handling | Raw files are read as distributed and canvases are written losslessly (PNG); Protocols A and B use no JPEG augmentation, as the SIDTD paper's models read the raw files. Every test is also reported with query and reference canvases re-encoded at quality 75 ("equalized") | SIDTD paper protocol; our audit | The raw SIDTD files carry a compression cue (genuine 92, altered 75), see Section 5; the equalized table shows whether a model relies on it. |
```

and append to Section 5 one paragraph: "From 10 October 2026 the cue is not removed in training (Protocols A and B follow the SIDTD paper, which reads raw files); it is measured instead, by the equalized test beside every result."

- [ ] **Step 2: Paper plan**

In `docs/paper-plan.md` Section 3 ("The three protocols"), retitle it "The four protocols" and replace the table rows so that A is the SIDTD template 10-fold anchor (train and test as in the paper's folds; reference per-fold median from training files; comparison: the paper's Table 5 composite-detection rows and our validated control), B the ten held-out-nationality groups (comparison: the control validated in A), C the FantasyID protocol (formerly B) and D the IDNet protocol (formerly C), keeping the existing C and D cell text otherwise. Copy the file to the vault path above.

- [ ] **Step 3: Commit both repositories**

```bash
git add docs/conventions.md docs/paper-plan.md
git commit -m "Docs: JPEG handling and the four protocols

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
cd /Users/hamza/Documents/GitHub/Thesis && git add "00-Vault/04-Writing/2026-10-10-refer-or-clear-paper-plan.md" && \
git commit -m "Refer or Clear plan: four protocols

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
