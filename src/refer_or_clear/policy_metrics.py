"""Small, tested policy helpers for the proposed research.

This module does not train a detector or reproduce any dataset results.
Callers must supply ONE predeclared forged representative per provenance
component to calibration. Raw repeated captures are not independent units.
Higher scores mean altered; referral is gate OR score > threshold.
Policy labels are 0=negative baseline and 1=altered. The negative baseline is
a mock control with no additional protocol edit for SIDTD-derived experiments,
or a bona-fide capture for FantasyID. It does not certify an issued document.
Legacy "genuine" keys remain available for compatibility. Prefer the neutral
n_negative, negative_referral, and gated_negative aliases for source results.
Native detection also exposes n_negative and f1_negative_class aliases; these
refer to the project's non-altered class, which has label 1 in the separate
bona-fide-positive DeepID convention, not to that convention's negative label.

DeepID-style compatibility helpers below are deliberately separate from that
policy. Detection uses an uncalibrated native bona-fide score and a fixed 0.5
threshold. Localization requires actual binary pixel annotations and binary
segmentation predictions; field boxes and Grad-CAM are not pixel ground truth.
These helpers do not reproduce a leaderboard cohort or the private aggregate.
"""
from __future__ import annotations
import math
from typing import Iterable


def _finite_score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Ungated scores must be finite; do not silently drop NaNs.")
    return value


def empirical_threshold(
    scores: Iterable[float], gates: Iterable[bool] | None = None, alpha: float = 0.05
) -> float:
    """Target empirical attack miss <= alpha; no population/shift guarantee."""
    values = list(scores)
    flags = list(gates) if gates is not None else [False] * len(values)
    if not values or len(flags) != len(values):
        raise ValueError("Need nonempty scores with one gate per calibration unit.")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1).")
    effective = [
        math.inf if gate else _finite_score(score)
        for score, gate in zip(values, flags, strict=True)
    ]
    effective.sort()
    rank = math.floor(alpha * len(effective)) + 1  # one-based rank
    cut = effective[rank - 1]
    # A gate remains a referral even when the selected cut is positive infinity.
    return math.inf if math.isinf(cut) else math.nextafter(cut, -math.inf)


def refer(score: float, threshold: float, gate: bool = False) -> bool:
    if math.isnan(float(threshold)):
        raise ValueError("Threshold must not be NaN.")
    return True if gate else _finite_score(score) > threshold


def confusion_and_rates(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float,
    gates: Iterable[bool] | None = None,
) -> dict:
    """Policy rates for 0=negative baseline and 1=altered.

    SIDTD negatives are baseline-unmodified mocks; FantasyID negatives are
    bona-fide captures. Neutral aliases and legacy genuine keys are equal.
    """
    y, s = list(labels), list(scores)
    flags = list(gates) if gates is not None else [False] * len(y)
    if not y or len(y) != len(s) or len(y) != len(flags):
        raise ValueError("Need equal-length, nonempty inputs.")
    if any(label not in (0, 1) for label in y):
        raise ValueError("Labels must be 0=negative baseline or 1=altered.")
    counts = dict(tp=0, fn=0, fp=0, tn=0,
                  gated_altered=0, gated_genuine=0)
    for label, score, gate in zip(y, s, flags, strict=True):
        decision = refer(score, threshold, gate)
        key = ("tp" if decision else "fn") if label == 1 else ("fp" if decision else "tn")
        counts[key] += 1
        if gate:
            counts["gated_altered" if label else "gated_genuine"] += 1
    na = counts["tp"] + counts["fn"]
    ng = counts["fp"] + counts["tn"]
    nr = counts["tp"] + counts["fp"]
    return {
        **counts, "n_altered": na, "n_genuine": ng, "n_negative": ng,
        "gated_negative": counts["gated_genuine"],
        "attack_miss": counts["fn"] / na if na else None,
        "recall": counts["tp"] / na if na else None,
        "genuine_referral": counts["fp"] / ng if ng else None,
        "negative_referral": counts["fp"] / ng if ng else None,
        "observed_referral_rate": nr / len(y),
        "observed_referral_precision": counts["tp"] / nr if nr else None,
    }


def prevalence_scenario(attack_miss: float, genuine_referral: float, p: float) -> dict:
    """A hypothetical deployment mixture, not measured deployment performance."""
    if not all(0 <= value <= 1 for value in (attack_miss, genuine_referral, p)):
        raise ValueError("Rates and prevalence must be in [0,1].")
    r = p * (1 - attack_miss) + (1 - p) * genuine_referral
    return {
        "assumed_prevalence": p,
        "referral_fraction": r,
        "referral_precision": p * (1 - attack_miss) / r if r else None,
        "altered_fraction_among_cleared": p * attack_miss / (1 - r) if r < 1 else None,
        "misses_per_10000": 10000 * p * attack_miss,
    }


def zero_miss_upper(n_independent: int, delta: float = 0.05) -> float:
    """One-sided exact upper bound after ZERO misses at an independently fixed policy.

    Requires independent Bernoulli trials for the declared population.
    This is not a post-selection threshold certificate or domain-shift guarantee.
    """
    if not isinstance(n_independent, int) or n_independent < 1 or not 0 < delta < 1:
        raise ValueError("Need a positive independent trial count and 0 < delta < 1.")
    return -math.expm1(math.log(delta) / n_independent)


def _binary_labels(labels: Iterable[int]) -> list[int]:
    values = list(labels)
    if not values or any(label not in (0, 1) for label in values):
        raise ValueError("Need nonempty binary labels, explicitly oriented as documented.")
    return [int(label) for label in values]


def _native_probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Native scores must be finite and in [0,1]; never drop invalid outputs.")
    return value


def deepid_detection_metrics(
    bona_fide_labels: Iterable[int],
    bona_fide_scores: Iterable[float],
    *,
    require_both_classes: bool = True,
) -> dict:
    """DeepID-style native-score detection, with 1=genuine and 0=altered.

    Score >= 0.5 predicts genuine, including exact ties. The public challenge
    description gives the threshold but not the detection tie inequality;
    this convention is explicit, not a claim of byte-exact scorer equivalence.
    Weighted F1 uses TRUE class supports, as sklearn average='weighted' does.
    Undefined per-class F1 is zero, matching sklearn zero_division=0.

    No referral gates, source thresholds, target fitting, or missing-score
    exclusions are applied. A complete native score is required for every
    declared cohort item. Use the separate policy functions for gate handling.
    Set require_both_classes=False only for labeled diagnostic subgroups;
    single-class cohorts do not constitute a challenge-style detection test.

    Sources: https://deepid-iccv.github.io/#leaderboard and
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    y = _binary_labels(bona_fide_labels)
    scores = [_native_probability(score) for score in bona_fide_scores]
    if len(y) != len(scores):
        raise ValueError("Need one native score for every label.")
    genuine = sum(y)
    altered = len(y) - genuine
    both = bool(genuine and altered)
    if require_both_classes and not both:
        raise ValueError("A challenge-style cohort must contain genuine and altered images.")
    counts = dict(tp=0, fn=0, fp=0, tn=0)  # bona fide is positive here
    for label, score in zip(y, scores, strict=True):
        predicted_genuine = score >= 0.5
        key = ("tp" if predicted_genuine else "fn") if label else ("fp" if predicted_genuine else "tn")
        counts[key] += 1
    tp, fn, fp, tn = (counts[key] for key in ("tp", "fn", "fp", "tn"))
    genuine_denominator = 2 * tp + fp + fn
    altered_denominator = 2 * tn + fp + fn
    f1_genuine = 2 * tp / genuine_denominator if genuine_denominator else 0.0
    f1_altered = 2 * tn / altered_denominator if altered_denominator else 0.0
    return {
        "n_images": len(y), "n_genuine": genuine, "n_negative": genuine,
        "n_altered": altered,
        "cohort_has_both_classes": both,
        "native_bona_fide_threshold": 0.5,
        "tie_rule": "bona_fide_score >= 0.5 predicts genuine",
        "bona_fide_positive_confusion": counts,
        "altered_positive_confusion": dict(tp=tn, fn=fp, fp=fn, tn=tp),
        "f1_genuine": f1_genuine,
        "f1_negative_class": f1_genuine,
        "f1_altered": f1_altered,
        "weighted_f1": (genuine * f1_genuine + altered * f1_altered) / len(y),
        "native_attack_miss": fp / altered if altered else None,
        "native_genuine_flag_rate": fn / genuine if genuine else None,
    }


def deepid_detection_from_suspicion(
    altered_labels: Iterable[int],
    suspicion_scores: Iterable[float],
    *,
    require_both_classes: bool = True,
) -> dict:
    """Adapter from this project's 1=altered labels and high=suspicious scores.

    The native bona-fide score is 1-suspicion. This inversion is not a
    calibration or threshold remapping. Supply sigmoid scores, not logits.
    """
    y = _binary_labels(altered_labels)
    scores = [_native_probability(score) for score in suspicion_scores]
    return deepid_detection_metrics(
        [1 - label for label in y], [1 - score for score in scores],
        require_both_classes=require_both_classes,
    )


def _binary_pixel_rows(mask: Iterable[Iterable[int]]) -> tuple[tuple[bool, ...], ...]:
    """Validate a nonempty rectangular 2D {0,1} mask, then cast to bool.

    PNG {0,255} storage must be explicitly decoded to {0,1} upstream. Never
    use bitwise inversion on integer masks: uint8 ~0 and ~1 are 255 and 254.
    Pure Python also accepts ordinary 2D NumPy arrays without requiring NumPy.
    """
    try:
        rows = tuple(tuple(row) for row in mask)
    except TypeError as exc:
        raise ValueError("Pixel masks must be rectangular 2D arrays.") from exc
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Pixel masks must be nonempty and rectangular.")
    output = []
    for row in rows:
        converted = []
        for value in row:
            if value not in (0, 1):
                raise ValueError("Pixel masks must contain only 0=altered and 1=genuine.")
            converted.append(bool(value))
        output.append(tuple(converted))
    return tuple(output)


def deepid_localization_metrics(
    bona_fide_labels: Iterable[int],
    predicted_masks: Iterable[Iterable[Iterable[int]]],
    ground_truth_masks: Iterable[Iterable[Iterable[int]]],
    *,
    ground_truth_kind: str,
    require_both_classes: bool = True,
) -> dict:
    """Optional DeepID-style metric for genuine binary pixel-mask evaluation.

    All masks use 0=altered, 1=genuine. The caller must explicitly declare
    ground_truth_kind='true_pixel_mask'. This declaration must be supported by
    the annotation audit; it cannot make field boxes or attribution maps true
    pixel annotations. Predictions must be binary segmentation outputs on the
    matching pixel grid. This helper adds no trained segmentation model.

    Genuine images score the genuine pixel class; altered images score the
    altered pixel class. Average per-image F1 within each image class, then
    equally average those TWO means. This is neither pooled-pixel F1 nor an
    image-count-weighted average. The public site's 1e-8 stabilizers are used.
    MCC is additionally averaged only over altered images; zero-denominator
    cases receive zero and their count is disclosed. Without both image
    classes, the combined localization F1 is unavailable, not renormalized.

    Sources: https://deepid-iccv.github.io/#leaderboard and
    https://publications.idiap.ch/attachments/papers/2025/Korshunov_ICCV_2025.pdf
    """
    if ground_truth_kind != "true_pixel_mask":
        raise ValueError("Localization requires audited true pixel masks, not field boxes or Grad-CAM.")
    y = _binary_labels(bona_fide_labels)
    predicted, truth = list(predicted_masks), list(ground_truth_masks)
    if len(y) != len(predicted) or len(y) != len(truth):
        raise ValueError("Need one prediction and ground-truth mask per image label.")
    genuine_count = sum(y)
    altered_count = len(y) - genuine_count
    if require_both_classes and not (genuine_count and altered_count):
        raise ValueError("Combined localization F1 requires genuine and altered images.")
    genuine_f1, altered_f1, attack_mcc, per_image = [], [], [], []
    degenerate_mcc = 0
    epsilon = 1e-8
    for index, (label, pred, gt) in enumerate(zip(y, predicted, truth, strict=True)):
        pred_rows, gt_rows = _binary_pixel_rows(pred), _binary_pixel_rows(gt)
        if (len(pred_rows), len(pred_rows[0])) != (len(gt_rows), len(gt_rows[0])):
            raise ValueError("Prediction and ground truth must have exactly the same pixel grid.")
        pred_values = tuple(value for row in pred_rows for value in row)
        gt_values = tuple(value for row in gt_rows for value in row)
        if label and not all(gt_values):
            raise ValueError("A genuine image must have an all-one ground-truth mask.")
        if not label and all(gt_values):
            raise ValueError("An altered image needs at least one annotated altered pixel.")
        if label:
            tp = sum(pred_values)
            fn, fp, tn = len(pred_values) - tp, 0, 0
            mcc, mcc_is_degenerate = None, False
        else:
            tp = fp = fn = tn = 0
            for p, g in zip(pred_values, gt_values, strict=True):
                if not p and not g:
                    tp += 1
                elif not p and g:
                    fp += 1
                elif p and not g:
                    fn += 1
                else:
                    tn += 1
            denominator_squared = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc_is_degenerate = denominator_squared == 0
            mcc = (tp * tn - fp * fn) / math.sqrt(denominator_squared) if denominator_squared else 0.0
            degenerate_mcc += int(mcc_is_degenerate)
            attack_mcc.append(mcc)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        (genuine_f1 if label else altered_f1).append(f1)
        per_image.append({
            "index": index, "bona_fide_label": label,
            "pixel_positive_class": "genuine" if label else "altered",
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "f1": f1, "attack_mcc": mcc,
            "attack_mcc_degenerate": mcc_is_degenerate,
        })
    mean_genuine = sum(genuine_f1) / len(genuine_f1) if genuine_f1 else None
    mean_altered = sum(altered_f1) / len(altered_f1) if altered_f1 else None
    return {
        "n_images": len(y), "n_genuine": genuine_count, "n_altered": altered_count,
        "cohort_has_both_classes": bool(genuine_count and altered_count),
        "mean_genuine_image_f1": mean_genuine,
        "mean_altered_image_f1": mean_altered,
        "balanced_image_class_f1": (mean_genuine + mean_altered) / 2
        if mean_genuine is not None and mean_altered is not None else None,
        "mean_attack_mcc": sum(attack_mcc) / len(attack_mcc) if attack_mcc else None,
        "n_attack_mcc_degenerate": degenerate_mcc,
        "n_attack_mcc_nondegenerate": altered_count - degenerate_mcc,
        "per_image": per_image,
    }


def _deepid_self_check() -> None:
    perfect = deepid_detection_metrics([0, 1], [0.1, 0.9])
    assert perfect["weighted_f1"] == 1.0
    assert perfect["altered_positive_confusion"] == dict(tp=1, fn=0, fp=0, tn=1)
    tie = deepid_detection_metrics([0, 1], [0.5, 0.5])
    assert tie["native_attack_miss"] == 1.0
    assert tie["native_genuine_flag_rate"] == 0.0
    assert math.isclose(tie["weighted_f1"], 1 / 3)
    boundary = deepid_detection_metrics(
        [0, 1, 1], [math.nextafter(0.5, -math.inf), 0.5, math.nextafter(0.5, math.inf)]
    )
    assert boundary["weighted_f1"] == 1.0
    converted = deepid_detection_from_suspicion([1, 0], [0.9, 0.1])
    assert converted == perfect
    assert deepid_detection_from_suspicion([1, 0], [0.5, 0.5]) == tie
    # A high weighted F1 can coexist with missing EVERY alteration.
    imbalanced = deepid_detection_metrics([0] + [1] * 9, [1.0] * 10)
    assert imbalanced["native_attack_miss"] == 1.0
    assert imbalanced["f1_altered"] == 0.0
    assert math.isclose(imbalanced["weighted_f1"], 0.9 * 18 / 19)
    single = deepid_detection_metrics([1, 1], [1.0, 1.0], require_both_classes=False)
    assert single["weighted_f1"] == 1.0 and not single["cohort_has_both_classes"]
    assert single["native_attack_miss"] is None

    one = ((1, 1), (1, 1))
    zero = ((0, 0), (0, 0))
    altered = ((0, 1), (1, 1))
    half = ((1, 1), (0, 0))
    kind = dict(ground_truth_kind="true_pixel_mask")
    exact = deepid_localization_metrics([1, 0], [one, altered], [one, altered], **kind)
    assert math.isclose(exact["balanced_image_class_f1"], 1.0, abs_tol=1e-7)
    assert exact["mean_attack_mcc"] == 1.0
    assert exact["n_attack_mcc_degenerate"] == 0
    boolean = deepid_localization_metrics(
        [1, 0],
        [tuple(tuple(bool(v) for v in row) for row in m) for m in (one, altered)],
        [one, altered], **kind,
    )
    assert boolean == exact
    # All-genuine pixel predictions score about 0.5, independent of image-class imbalance.
    all_genuine = deepid_localization_metrics([1] * 9 + [0], [one] * 10, [one] * 9 + [altered], **kind)
    assert math.isclose(all_genuine["balanced_image_class_f1"], 0.5, abs_tol=1e-7)
    assert all_genuine["mean_altered_image_f1"] == 0.0
    assert all_genuine["n_attack_mcc_degenerate"] == 1
    assert all_genuine["mean_attack_mcc"] == 0.0
    # On an authentic image, half the pixels incorrectly flagged gives F1=2/3.
    partial = deepid_localization_metrics([1, 0], [half, one], [one, altered], **kind)
    assert math.isclose(partial["mean_genuine_image_f1"], 2 / 3, abs_tol=1e-7)
    assert math.isclose(partial["balanced_image_class_f1"], 1 / 3, abs_tol=1e-7)
    # A quarter-image alteration predicted everywhere gives attack F1=2a/(1+a)=0.4.
    everywhere = deepid_localization_metrics([1, 0], [zero, zero], [one, altered], **kind)
    assert math.isclose(everywhere["mean_altered_image_f1"], 0.4, abs_tol=1e-7)
    assert math.isclose(everywhere["balanced_image_class_f1"], 0.2, abs_tol=1e-7)
    inverse = ((1, 0), (0, 0))
    reversed_mask = deepid_localization_metrics([1, 0], [one, inverse], [one, altered], **kind)
    assert reversed_mask["mean_attack_mcc"] == -1.0
    full_alteration = deepid_localization_metrics([1, 0], [one, zero], [one, zero], **kind)
    assert math.isclose(full_alteration["mean_altered_image_f1"], 1.0, abs_tol=1e-7)
    assert full_alteration["n_attack_mcc_degenerate"] == 1
    missing_class = deepid_localization_metrics([1], [one], [one], require_both_classes=False, **kind)
    assert missing_class["balanced_image_class_f1"] is None
    assert missing_class["mean_attack_mcc"] is None

    invalid_calls = [
        lambda: deepid_detection_metrics([], []),
        lambda: deepid_detection_metrics([0, 1], [0.1]),
        lambda: deepid_detection_metrics([0, 2], [0.1, 0.9]),
        lambda: deepid_detection_metrics([1], [0.9]),
        lambda: deepid_localization_metrics([1, 0], [one, altered], [one, altered], ground_truth_kind="field_box"),
        lambda: deepid_localization_metrics([1, 0], [one, altered], [altered, altered], **kind),
        lambda: deepid_localization_metrics([1, 0], [one, altered], [one, one], **kind),
        lambda: deepid_localization_metrics([1, 0], [one, altered], [one], **kind),
        lambda: deepid_localization_metrics([1, 0], [one, ((0, 1),)], [one, altered], **kind),
        lambda: deepid_localization_metrics([1], [one], [one], **kind),
    ]
    invalid_calls.extend(
        lambda value=value: deepid_detection_metrics([0, 1], [value, 0.9])
        for value in (math.nan, math.inf, -math.inf, -0.01, 1.01)
    )
    invalid_calls.extend(
        lambda mask=mask: deepid_localization_metrics([1, 0], [one, mask], [one, altered], **kind)
        for mask in ([], [0, 1], ((0,), (1, 1)), ((0, 255), (1, 1)),
                     ((0.2, 1), (1, 1)), ((math.nan, 1), (1, 1)))
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid native scores, masks or cohorts must fail explicitly.")
    # Optional independent references; neither package is a runtime dependency.
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        pass
    else:
        for labels, scores in (([0, 1], [0.5, 0.5]),
                               ([0] + [1] * 9, [1.0] * 10),
                               ([0, 0, 1, 1], [0.1, 0.9, 0.1, 0.9]),
                               ([1, 1], [1.0, 1.0])):
            measured = deepid_detection_metrics(labels, scores, require_both_classes=False)
            reference = f1_score(labels, [int(s >= 0.5) for s in scores],
                                 average="weighted", labels=[0, 1], zero_division=0)
            assert math.isclose(measured["weighted_f1"], reference)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        uint8 = deepid_localization_metrics(
            [1, 0], [np.array(one, dtype=np.uint8), np.array(altered, dtype=np.uint8)],
            [np.array(one, dtype=np.uint8), np.array(altered, dtype=np.uint8)], **kind,
        )
        assert uint8 == exact


def self_check() -> None:
    baseline = confusion_and_rates([1, 1, 0, 0], [.9, .1, .8, .2], .5)
    assert (baseline["tp"], baseline["fn"], baseline["fp"], baseline["tn"]) == (1, 1, 1, 1)
    assert baseline["attack_miss"] == baseline["genuine_referral"] == .5
    gated = confusion_and_rates([1, 1, 0, 0], [.9, math.nan, .8, .2],
                                .5, [False, True, False, False])
    assert gated["attack_miss"] == 0 and gated["n_altered"] == 2
    assert gated["genuine_referral"] == .5 and gated["gated_altered"] == 1
    cal = [i / 20 for i in range(1, 21)]
    t = empirical_threshold(cal)
    assert sum(not refer(score, t) for score in cal) == 1
    tied = [.1] * 20
    assert all(refer(score, empirical_threshold(tied)) for score in tied)
    assert math.isinf(empirical_threshold([math.nan], [True]))
    assert refer(math.nan, math.inf, True)
    try:
        empirical_threshold([math.nan])
    except ValueError:
        pass
    else:
        raise AssertionError("Ungated NaN should fail.")
    sc = prevalence_scenario(.05, .01, .01)
    assert math.isclose(sc["referral_fraction"], .0194)
    assert math.isclose(sc["referral_precision"], .4896907216494845)
    assert zero_miss_upper(59) < .05 < zero_miss_upper(58)
    assert zero_miss_upper(299) < .01 < zero_miss_upper(298)
    _deepid_self_check()


if __name__ == "__main__":
    self_check()
    print("Policy and DeepID-style metric correctness checks passed. No dataset experiments were run.")
