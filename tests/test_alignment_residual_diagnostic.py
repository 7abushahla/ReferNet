"""Synthetic checks for the training-free alignment-residual diagnostic."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from alignment_residual_diagnostic import auc, score_pair  # noqa: E402


def _textured(seed):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((512, 800), np.float32) * 255., (0, 0), 1.5)


def test_pasted_field_shift_is_recovered_and_genuine_is_near_zero():
    reference = _textured(0)
    noise = np.random.default_rng(1).normal(0, 5, reference.shape).astype(np.float32)
    genuine = reference + noise
    zone = np.zeros((512, 800), np.uint8)
    zone[100:140, 100:300] = 1
    zone[200:240, 100:300] = 1
    overlap = np.ones((512, 800), np.uint8)
    copied = genuine.copy()
    matrix = np.float32([[1, 0, 6], [0, 1, -4]])
    copied[200:240, 100:300] = cv2.warpAffine(genuine, matrix, (800, 512), borderMode=cv2.BORDER_REPLICATE)[200:240, 100:300]
    clean, boxes = score_pair(genuine, reference, zone, overlap)
    pasted, pasted_boxes = score_pair(copied, reference, zone, overlap)
    assert clean["n_boxes"] == 2 and len(boxes) == 2
    assert clean["shift_max"] < 1.
    assert abs(pasted["shift_max"] - np.hypot(6, 4)) < 1.
    assert pasted["resid_max"] > clean["resid_max"]
    assert pasted["resid_gain"] > clean["resid_gain"]
    assert max(pasted_boxes, key=lambda r: r["shift"])["y"] == 200


def test_boxes_outside_overlap_are_skipped():
    reference = _textured(2)
    zone = np.zeros((512, 800), np.uint8)
    zone[100:140, 100:300] = 1
    overlap = np.zeros((512, 800), np.uint8)
    scores, boxes = score_pair(reference, reference, zone, overlap)
    assert scores["n_boxes"] == 0 and boxes == []


def test_auc_handles_ties_and_degenerate_classes():
    assert auc([0.1, 0.4, 0.3, 0.9], [0, 0, 1, 1]) == 0.75
    assert auc([0.5, 0.5], [0, 1]) == 0.5
    assert auc([0.2, 0.3], [1, 1]) is None
