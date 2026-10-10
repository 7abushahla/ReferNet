"""Checks for metrics.py on synthetic scores and toy masks. Plain asserts; run with

    PYTHONPATH=src python tests/test_metrics.py

Scores are bona fide probabilities, labels 1 = altered (see metrics.py docstring).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear import metrics as M  # noqa: E402
from refer_or_clear import policy_metrics as PM  # noqa: E402


def separable():
    s = np.concatenate([np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.4, 50)])
    y = np.concatenate([np.zeros(50, int), np.ones(50, int)])
    return s, y


def overlapping(seed=0):
    rng = np.random.RandomState(seed)
    s = np.concatenate([rng.beta(4, 2, 300), rng.beta(2, 4, 200)])
    y = np.concatenate([np.zeros(300, int), np.ones(200, int)])
    return s, y


def check_auc_eer():
    s, y = separable()
    assert M.auc(s, y) == 1.0
    assert M.eer(s, y) == 0.0
    assert M.auc(1 - s, y) == 0.0, "flipping the orientation must flip the AUC"
    s, y = overlapping()
    a, e = M.auc(s, y), M.eer(s, y)
    assert 0.5 < a < 1.0 and 0.0 < e < 0.5, (a, e)
    t = M.eer_threshold(s, y)
    ap, bp = M.apcer_bpcer(s, y, t)
    assert abs(ap - bp) < 0.03 and math.isclose(e, (ap + bp) / 2)
    rng = np.random.RandomState(1)
    r = rng.rand(2000)
    yr = rng.randint(0, 2, 2000)
    assert abs(M.auc(r, yr) - 0.5) < 0.05
    assert M.auc(np.ones(3), [0, 0, 0]) is None and M.eer(np.ones(3), [1, 1, 1]) is None


def check_far_threshold():
    for s, y in (separable(), overlapping(), overlapping(3)):
        for target in (0.0, 0.01, 0.05, 0.10, 0.5):
            t = M.far_threshold(s, y, target)
            ap, bp = M.apcer_bpcer(s, y, t)
            assert ap <= target + 1e-12, (target, ap)
            assert 0 <= bp <= 1
            cands = M.threshold_candidates(s)
            lower = cands[cands < t]
            for c in lower[-3:]:  # every lower candidate would overshoot the budget
                assert M.apcer_bpcer(s, y, c)[0] > target, (target, c, t)
            assert M.transported(s, y, t) == {"miss": ap, "referral": bp}
            assert math.isclose(M.hter(s, y, t), (ap + bp) / 2)
    s, y = separable()
    t0 = M.far_threshold(s, y, 0.0)
    assert 0.4 < t0 < 0.6 and M.apcer_bpcer(s, y, t0) == (0.0, 0.0)
    # tie rule: a score on the threshold is accepted as bona fide
    assert M.apcer_bpcer([0.5, 0.5], [1, 0], 0.5) == (1.0, 0.0)
    # target 1 accepts everything at the lowest observed score
    assert M.far_threshold(s, y, 1.0) == s.min()
    assert M.far_threshold(np.ones(2), [0, 0], 0.05) is None
    d = M.bpcer_at_apcer(*overlapping())
    assert list(d) == [0.01, 0.05, 0.10] and d[0.01] >= d[0.05] >= d[0.10]
    assert M.transported(s, y, None) == {"miss": None, "referral": None}


def check_deepid():
    s = np.array([0.9, 0.5, 0.49, 0.1, 0.7, 0.2])
    y = np.array([0, 0, 1, 1, 1, 0])  # 0.5 counts as genuine; 0.7 altered is a miss; 0.2 genuine is referred
    assert math.isclose(M.deepid_acc(s, y), 4 / 6)
    ref = PM.deepid_detection_metrics(1 - y, s)  # explicit conversion to the bona fide-positive helper
    assert math.isclose(M.deepid_f1(s, y), ref["weighted_f1"])
    s, y = separable()
    assert M.deepid_f1(s, y) == 1.0 and M.deepid_acc(s, y) == 1.0
    assert M.deepid_f1(np.ones(4), [1, 1, 1, 1]) == 0.0  # everything accepted, all altered: zero recall
    assert M.deepid_f1([], []) is None


def check_localization():
    gt = np.zeros((4, 4), bool)
    gt[1:3, 1:3] = True  # altered quarter
    valid = np.ones((4, 4), bool)
    exact = gt.astype(np.float32)
    everywhere = np.ones((4, 4), np.float32)
    nothing = np.zeros((4, 4), np.float32)
    # altered image: F1 of the altered class
    assert math.isclose(M.localization_f1(exact, gt, valid), 1.0, abs_tol=1e-6)
    assert math.isclose(M.localization_f1(everywhere, gt, valid), 0.4, abs_tol=1e-6)  # 2a/(1+a), a=1/4
    assert math.isclose(M.localization_f1(nothing, gt, valid), 0.0, abs_tol=1e-6)
    # genuine image: F1 of the clean class
    clean = np.zeros((4, 4), bool)
    assert math.isclose(M.localization_f1(nothing, clean, valid), 1.0, abs_tol=1e-6)
    assert math.isclose(M.localization_f1(everywhere, clean, valid), 0.0, abs_tol=1e-6)
    half = np.zeros((4, 4), np.float32)
    half[:2] = 1
    assert math.isclose(M.localization_f1(half, clean, valid), 2 / 3, abs_tol=1e-6)
    # polarity matches the audited DeepID helper (its masks use 1 = genuine)
    ref = PM.deepid_localization_metrics([1, 0], [(1 - half).astype(int), (1 - gt).astype(int)],
                                         [np.ones((4, 4), int), (1 - gt).astype(int)],
                                         ground_truth_kind="true_pixel_mask")
    ours = M.per_image_localization_f1(np.stack([half, exact]), np.stack([clean, gt]), np.stack([valid, valid]))
    assert math.isclose(ours[0], ref["mean_genuine_image_f1"], abs_tol=1e-6)
    assert math.isclose(ours[1], ref["mean_altered_image_f1"], abs_tol=1e-6)
    # labels override the mask heuristic; invalid pixels are ignored
    f = M.per_image_localization_f1(np.stack([nothing, nothing]), np.stack([clean, clean]), None, labels=[0, 1])
    assert math.isclose(f[0], 1.0, abs_tol=1e-6) and math.isclose(f[1], 0.0, abs_tol=1e-6)
    v = valid.copy()
    v[0] = False
    assert math.isclose(M.localization_f1(half, clean, v), 2 * 8 / (2 * 8 + 4), abs_tol=1e-6)
    assert M.localization_f1(np.zeros((0, 4, 4)), np.zeros((0, 4, 4)), None) is None


def check_fp_area():
    zone = np.zeros((4, 4), bool)
    zone[:, :2] = True  # left half is a field
    valid = np.ones((4, 4), bool)
    pred = np.zeros((4, 4), np.float32)
    pred[0, :] = 1  # top row flagged: 2 px in zone, 2 px out of 16
    inside, outside = M.fp_map_area(pred, zone, valid)
    assert math.isclose(inside, 2 / 16) and math.isclose(outside, 2 / 16)
    total = (pred >= 0.5).mean()
    assert math.isclose(inside + outside, total)
    everywhere = np.ones((4, 4), np.float32)
    inside, outside = M.fp_map_area(everywhere, zone, valid)
    assert math.isclose(inside + outside, 1.0) and math.isclose(inside, 0.5)
    v = valid.copy()
    v[:, 3] = False  # 12 valid px; top row now 2 in zone, 1 out
    inside, outside = M.fp_map_area(pred, zone, v)
    assert math.isclose(inside, 2 / 12) and math.isclose(outside, 1 / 12)
    assert M.fp_map_area(np.zeros((0, 4, 4)), np.zeros((0, 4, 4)), None) == (None, None)


if __name__ == "__main__":
    for check in (check_auc_eer, check_far_threshold, check_deepid, check_localization, check_fp_area):
        check()
        print(f"[ok] {check.__name__}")
    print("test_metrics: all checks passed")
