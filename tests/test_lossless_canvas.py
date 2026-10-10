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
