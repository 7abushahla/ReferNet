import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from dump_inputs import read_masks, read_rgb, write_masks, write_rgb  # noqa: E402


def test_rgb_png_roundtrip_is_exact(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 256, (512, 800, 3), dtype=np.uint8)
    write_rgb(tmp_path / "a.png", rgb)
    assert np.array_equal(read_rgb(tmp_path / "a.png"), rgb)


def test_masks_png_roundtrip_is_exact(tmp_path):
    g = np.random.default_rng(1)
    zone, overlap = (g.random((512, 800)) > 0.5).astype(np.uint8), (g.random((512, 800)) > 0.3).astype(np.uint8)
    write_masks(tmp_path / "m.png", zone, overlap)
    z, o = read_masks(tmp_path / "m.png")
    assert np.array_equal(z, zone) and np.array_equal(o, overlap)
    from dump_inputs import read_rgb as rr
    m = rr(tmp_path / "m.png")
    assert set(np.unique(m[..., 0])) <= {0, 255} and set(np.unique(m[..., 1])) <= {0, 255} and not m[..., 2].any()


def test_sources_name_every_design():
    d = json.loads((ROOT / "demo/sources.json").read_text())
    assert d["quality"] == 75
    for k in ("registration", "raw_root", "source", "diag_align", "runs", "weights"):
        assert d["a"][k], k
    assert sorted(d["c"]) == ["ALB", "AZ", "ESP", "FIN", "GRC", "NV", "SD", "WV"]


def test_a_degenerate_cross_design_fit_counts_as_no_fit():
    """The paper's registration treats LinAlgError, cv2.error and ValueError as 'registration failed'."""
    from dump_inputs import fits_or_false

    def singular():
        raise np.linalg.LinAlgError("Singular matrix")
    assert fits_or_false(singular) is False
    assert fits_or_false(lambda: True) is True
