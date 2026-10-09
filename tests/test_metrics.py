"""Sanity checks for metrics.py once implemented: perfect scores give AUC 1 and EER 0; random scores give
AUC near 0.5; referral rate at miss 0 equals 1 when classes overlap fully."""
import pytest


@pytest.mark.skip(reason="metrics not implemented yet")
def test_auc_perfect():
    pass
