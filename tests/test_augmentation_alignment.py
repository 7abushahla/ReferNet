"""Image geometry and dense targets must remain in the same coordinate system."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.data.dataset import PairDataset


class IdentityGeometry(PairDataset):
    def _geometry(self, height, width):
        return np.float32([[1, 0, 0], [0, 1, 0]]), (slice(0, height), slice(0, width))


def test_query_jitter_moves_target_and_valid_region():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        mask = np.zeros((16, 20), np.uint8)
        mask[5:9, 6:11] = 255
        image = np.repeat(mask[..., None], 3, axis=2)
        cv2.imwrite(str(root / "query.png"), image)
        cv2.imwrite(str(root / "mask.png"), mask)
        pair = dict(path=str(root / "query.png"), ref_path=str(root / "query.png"),
                    mask_path=str(root / "mask.png"), zone_path=None, valid_path=None,
                    label=1, wrong_design=False, image_id="synthetic", design_id="design",
                    subset="", attack_family="synthetic")
        dataset = IdentityGeometry([pair], True, crop=None, jpeg=None, jitter_p=1,
                                   jitter_px=2, photometric=False)
        with patch("refer_or_clear.data.dataset.random.randint", side_effect=[2, -1]):
            item = dataset[0]
        expected = cv2.warpAffine(mask, np.float32([[1, 0, 2], [0, 1, -1]]), (20, 16),
                                  flags=cv2.INTER_NEAREST)
        assert torch.equal(item["mask"][0] > 0, torch.from_numpy(expected > 0)), "jitter left the target behind"
        assert torch.equal(item["x"][0] > 0, item["mask"][0] > 0)
        assert not bool(item["valid"][0, :, :2].any())
        assert not bool(item["valid"][0, -1].any())
        assert torch.equal(item["x"][4] > 0, torch.from_numpy(mask > 0)), "reference moved with query jitter"


if __name__ == "__main__":
    test_query_jitter_moves_target_and_valid_region()
    print("ok: query jitter preserves target alignment")
