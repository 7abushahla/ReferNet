import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.data import direct_reference as bridge
from refer_or_clear.data.reencode import reencoded_queries


def _write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_reencodes_only_raw_root_color_queries_and_restores(tmp_path):
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    query = _write(tmp_path / "raw/sidtd/q.png", noise)
    product = _write(tmp_path / "products/median.png", noise)
    mask = _write(tmp_path / "raw/sidtd/mask.png", np.full((64, 96), 255, np.uint8))
    original = bridge._image
    with reencoded_queries(bridge, tmp_path / "raw", 75) as treatment:
        assert treatment["quality"] == 75 and treatment["kind"] == "query_jpeg_reencode"
        changed = bridge._image(query, cv2.IMREAD_COLOR)
        untouched = bridge._image(product, cv2.IMREAD_COLOR)
        gray = bridge._image(mask, cv2.IMREAD_GRAYSCALE)
    assert bridge._image is original
    assert changed.shape == noise.shape and not np.array_equal(changed, noise)
    assert np.array_equal(untouched, noise) and np.array_equal(gray, np.full((64, 96), 255, np.uint8))
    assert np.array_equal(bridge._image(query, cv2.IMREAD_COLOR), noise)


@pytest.mark.parametrize("quality", [0, 101, 75.0, "75"])
def test_invalid_quality_rejected(tmp_path, quality):
    with pytest.raises(ValueError):
        with reencoded_queries(bridge, tmp_path, quality):
            pass


def test_decoder_restored_after_error(tmp_path):
    original = bridge._image
    with pytest.raises(RuntimeError):
        with reencoded_queries(bridge, tmp_path, 75):
            raise RuntimeError("inside")
    assert bridge._image is original
