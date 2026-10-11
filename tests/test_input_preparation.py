"""Direct resize geometry and safeguards against mixed preprocessing caches."""
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from refer_or_clear.data.rectify import to_canvas, rect_to_canvas
import prepare_data


def test_direct_resize_pixels_and_mask_coordinates():
    source = np.random.default_rng(42).integers(0, 256, (256, 200, 3), dtype=np.uint8)
    pixels, valid, transform = to_canvas(source, mode="direct")
    assert np.array_equal(pixels, cv2.resize(source, (800, 512), interpolation=cv2.INTER_AREA))
    assert valid.shape == (512, 800) and (valid == 255).all()
    np.testing.assert_array_equal(transform, [[4., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    assert rect_to_canvas(10, 20, 30, 40, transform) == (40, 40, 160, 120)
    letterboxed, letterbox_valid, _ = to_canvas(source)
    assert not np.array_equal(letterboxed, pixels)
    assert (letterbox_valid == 0).any()


def test_direct_build_is_lossless_and_rejects_registration():
    args = prepare_data.parse_args(["--input-mode", "direct", "--no-register", "--out", "data/direct", "--anchor-fold", "0"])
    assert args.input_mode == "direct" and args.no_register
    assert args.anchor_fold == 0
    try:
        prepare_data.parse_args(["--input-mode", "direct", "--out", "data/direct"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("direct resize silently enabled registration")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "source.png"), source)
        _, transform = prepare_data._canvas_job(("sample", root / "source.png", root / "output.png",
                                                root / "valid.png", None, "direct"))
        assert np.array_equal(cv2.imread(str(root / "output.png")), to_canvas(source, mode="direct")[0])
        assert not (root / "valid.png").exists()
        assert rect_to_canvas(0, 0, 128, 128, np.array(transform)) == (0, 0, 800, 512)


def test_cache_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "canvases").mkdir()
        (root / "canvases/sample.png").touch()
        try:
            prepare_data.check_preprocessing_cache(root, "direct", True, False)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown existing canvases accepted for direct resize")
        prepare_data.check_preprocessing_cache(root, "direct", True, True)


if __name__ == "__main__":
    test_direct_resize_pixels_and_mask_coordinates()
    test_direct_build_is_lossless_and_rejects_registration()
    test_cache_mismatch_is_rejected()
    print("ok: 3 input preparation tests")
