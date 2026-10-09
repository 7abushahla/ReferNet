"""Metrics (DESIGN.md 4.5, Table 3 of the proposal).

Image level : AUC, EER, APCER at BPCER 1 and 5 percent, HTER (ISO/IEC 30107-3), accuracy (SIDTD).
Mask level  : pixel F1 and IoU at a fixed threshold.
Operator    : referral rate R(t) at miss rate M(t) of 1 and 5 percent, prior p = 0.044.
"""


def auc(scores, labels):
    raise NotImplementedError


def eer(scores, labels):
    raise NotImplementedError


def apcer_at_bpcer(scores, labels, bpcer):
    raise NotImplementedError


def hter(scores, labels, threshold):
    raise NotImplementedError


def referral_rate_at_miss(scores, labels, miss, prior=0.044):
    raise NotImplementedError
