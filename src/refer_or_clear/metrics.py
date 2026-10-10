"""Image-level and map-level metrics (docs/conventions.md Section 1; implementation plan Section 4).

Orientation, stated once: `scores` are BONA FIDE probabilities (1 = genuine, DeepID export convention),
`labels` are this project's labels (1 = altered, 0 = genuine). A sample is accepted as bona fide when
score >= t (bob tie rule). APCER = fraction of altered accepted (the miss rate); BPCER = fraction of
genuine referred (the referral rate). Pure numpy and sklearn; the bob rules are reimplemented, not
imported. Map metrics take altered-probability maps and {0, 1} masks on the same pixel grid.
Functions return None where a rate is undefined because a class is absent.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-8  # DeepID stabiliser for the per-image F1


def _arrays(scores, labels):
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ in length")
    if s.size and not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if s.size and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (genuine) or 1 (altered)")
    return s, y


def threshold_candidates(scores) -> np.ndarray:
    """Sorted observed scores, the midpoints between neighbours, and one value just above the maximum."""
    u = np.unique(np.asarray(scores, dtype=np.float64).ravel())
    if u.size == 0:
        return u
    mids = (u[:-1] + u[1:]) / 2
    return np.unique(np.concatenate([u, mids, [np.nextafter(u[-1], np.inf)]]))


def auc(scores, labels):
    """ROC AUC with bona fide as the positive class; None unless both classes are present."""
    s, y = _arrays(scores, labels)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return None
    return float(roc_auc_score(1 - y, s))


def apcer_bpcer(scores, labels, t):
    """(APCER, BPCER) at acceptance threshold t: miss rate of altered, referral rate of genuine."""
    s, y = _arrays(scores, labels)
    accepted = s >= float(t)
    alt, gen = y == 1, y == 0
    apcer = float(accepted[alt].mean()) if alt.any() else None
    bpcer = float((~accepted[gen]).mean()) if gen.any() else None
    return apcer, bpcer


def _rates_at(s, y, cands):
    """APCER and BPCER for every candidate threshold, vectorised."""
    acc = s[None, :] >= cands[:, None]
    alt, gen = y == 1, y == 0
    apcer = acc[:, alt].mean(axis=1) if alt.any() else np.full(len(cands), np.nan)
    bpcer = (~acc[:, gen]).mean(axis=1) if gen.any() else np.full(len(cands), np.nan)
    return apcer, bpcer


def eer_threshold(scores, labels):
    """Observed-score threshold (bob eer_threshold rule) closest to APCER = BPCER; None without both classes."""
    s, y = _arrays(scores, labels)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return None
    cands = threshold_candidates(s)
    apcer, bpcer = _rates_at(s, y, cands)
    return float(cands[int(np.argmin(np.abs(apcer - bpcer)))])


def eer(scores, labels):
    """Equal error rate by the midpoint rule, (APCER + BPCER) / 2 at eer_threshold; never an operating point."""
    t = eer_threshold(scores, labels)
    if t is None:
        return None
    a, b = apcer_bpcer(scores, labels, t)
    return (a + b) / 2


def far_threshold(scores, labels, target_apcer):
    """bob far_threshold rule: the LOWEST acceptance threshold whose realised APCER is at most the target.

    Searched over observed scores and their midpoints; a score on the threshold is accepted. None if no altered
    sample is present (the constraint is undefined).
    """
    s, y = _arrays(scores, labels)
    if (y == 1).sum() == 0:
        return None
    if not 0 <= target_apcer <= 1:
        raise ValueError("target_apcer must be in [0, 1]")
    cands = threshold_candidates(s)
    apcer, _ = _rates_at(s, y, cands)
    ok = np.flatnonzero(apcer <= target_apcer + 1e-12)
    return float(cands[ok[0]])


def hter(scores, labels, t):
    """Half total error rate (APCER + BPCER) / 2 at a frozen threshold t; None if a class is absent."""
    a, b = apcer_bpcer(scores, labels, t)
    return None if a is None or b is None else (a + b) / 2


def bpcer_at_apcer(scores, labels, targets=(0.01, 0.05, 0.10)):
    """Descriptive BPCER at each APCER budget, threshold chosen on the SAME set; {target: bpcer or None}."""
    out = {}
    for target in targets:
        t = far_threshold(scores, labels, target)
        out[target] = None if t is None else apcer_bpcer(scores, labels, t)[1]
    return out


def transported(scores, labels, t):
    """Rates under a threshold frozen elsewhere: {'miss': APCER, 'referral': BPCER} (None when undefined)."""
    if t is None:
        return {"miss": None, "referral": None}
    a, b = apcer_bpcer(scores, labels, t)
    return {"miss": a, "referral": b}


def deepid_acc(scores, labels):
    """Accuracy at the fixed bona fide threshold 0.5 (score >= 0.5 predicts genuine)."""
    s, y = _arrays(scores, labels)
    if s.size == 0:
        return None
    pred_altered = (s < 0.5).astype(int)
    return float((pred_altered == y).mean())


def deepid_f1(scores, labels):
    """Weighted F1 at 0.5 as the DeepID detection rows (sklearn average='weighted', zero_division=0)."""
    s, y = _arrays(scores, labels)
    if s.size == 0:
        return None
    pred_altered = (s < 0.5).astype(int)
    return float(f1_score(y, pred_altered, average="weighted", labels=[0, 1], zero_division=0))


def _maps(pred_maps, gt_masks, valid):
    p = np.asarray(pred_maps, dtype=np.float32)
    g = np.asarray(gt_masks).astype(bool)
    v = np.ones_like(g) if valid is None else np.asarray(valid).astype(bool)
    if p.ndim == 2:
        p, g, v = p[None], g[None], v[None]
    if not (p.shape == g.shape == v.shape):
        raise ValueError(f"pred {p.shape}, gt {g.shape} and valid {v.shape} must share one grid")
    return p, g, v


def per_image_localization_f1(pred_maps, gt_masks, valid=None, thr=0.5, labels=None):
    """Per-image F1 with DeepID polarity: altered class on altered images, clean class on genuine images.

    An image counts as altered when `labels` says so, else when its mask has any altered pixel inside `valid`.
    """
    p, g, v = _maps(pred_maps, gt_masks, valid)
    pred = (p >= thr) & v
    gt = g & v
    out = np.zeros(len(p))
    for i in range(len(p)):
        altered = bool(labels[i]) if labels is not None else bool(gt[i].any())
        if altered:
            tp = float((pred[i] & gt[i]).sum())
            fp = float((pred[i] & ~gt[i]).sum())
            fn = float((~pred[i] & gt[i]).sum())
        else:  # clean class: predicted clean against an all-clean truth, inside valid
            tp = float((~pred[i] & v[i]).sum())
            fp = 0.0
            fn = float(pred[i].sum())
        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        out[i] = 2 * prec * rec / (prec + rec + EPS)
    return out


def localization_f1(pred_maps, gt_masks, valid=None, thr=0.5, labels=None):
    """Mean over images of per_image_localization_f1; None on an empty set."""
    f = per_image_localization_f1(pred_maps, gt_masks, valid, thr, labels)
    return float(f.mean()) if f.size else None


def per_image_fp_area(pred_maps, zones, valid=None, thr=0.5):
    """Per image, the fraction of the valid area predicted altered inside the zone and outside it (sums to total)."""
    p, z, v = _maps(pred_maps, zones, valid)
    if len(p) == 0:
        return np.zeros(0), np.zeros(0)
    pred = (p >= thr) & v
    n_valid = np.maximum(v.reshape(len(p), -1).sum(axis=1), 1).astype(np.float64)
    inside = (pred & z).reshape(len(p), -1).sum(axis=1) / n_valid
    outside = (pred & ~z).reshape(len(p), -1).sum(axis=1) / n_valid
    return inside, outside


def fp_map_area(pred_maps, zones, valid=None, thr=0.5):
    """On genuine images: mean predicted-altered area fraction (inside_zone, outside_zone); (None, None) if empty."""
    inside, outside = per_image_fp_area(pred_maps, zones, valid, thr)
    if inside.size == 0:
        return None, None
    return float(inside.mean()), float(outside.mean())
