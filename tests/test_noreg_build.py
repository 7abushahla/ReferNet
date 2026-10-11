"""Letterbox-only data build (no alignment warp) and sweeps on a different data dir: python tests/test_noreg_build.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_sweep as MS  # noqa: E402
import prepare_data as PD  # noqa: E402


def test_prepare_data_flags():
    a = PD.parse_args(["--no-register", "--out", "data/processed_noreg", "--datasets", "sidtd"])
    assert a.no_register and a.out == "data/processed_noreg" and a.datasets == "sidtd"
    a = PD.parse_args([])
    assert not a.no_register and a.out is None


def test_identity_registration_rows():
    rec = pd.DataFrame([dict(image_id="a", design_id="x/d"), dict(image_id="b", design_id="x/d")])
    reg = PD.identity_registration(rec)
    assert list(reg.columns) == ["design_id", "image_id", "applied", "inliers", "rot_deg", "scale", "shift_px"]
    assert (~reg.applied).all() and (reg.scale == 1.0).all() and (reg.shift_px == 0.0).all()


def test_sweep_data_dir():
    d = tempfile.mkdtemp()
    MS.write_sweep(["--protocols", "A", "--archs", "changerex_r18", "--fusions", "single_noex", "--references", "median",
                    "--inits", "levir", "--seeds", "42", "--folds", "0", "--name", "t", "--out", d,
                    "--data-dir", "data/processed_noreg", "--suffix=-noreg"])
    text = (Path(d) / "A-cex-single_noex-median-levir-f0-s42.sh").read_text()
    assert "rsync -a --delete data/processed_noreg/" in text and "data/processed_noreg/splits/*.csv" in text
    assert 'RUN_DIR="results/runs/A-cex-single_noex-median-levir-f0-s42-noreg"' in text
    d2 = tempfile.mkdtemp()
    MS.write_sweep(["--protocols", "A", "--archs", "changerex_r18", "--fusions", "single_noex", "--references", "median",
                    "--inits", "levir", "--seeds", "42", "--folds", "0", "--name", "t", "--out", d2])
    assert "rsync -a --delete data/processed/" in (Path(d2) / "A-cex-single_noex-median-levir-f0-s42.sh").read_text()


if __name__ == "__main__":
    test_prepare_data_flags()
    test_identity_registration_rows()
    test_sweep_data_dir()
    print("ok")
