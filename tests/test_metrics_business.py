"""Unit tests for business metrics and drift helpers."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    evaluate_classification,
    expected_financial_loss,
    find_best_threshold,
)
from src.monitoring.drift import (
    detect_feature_drift,
    detect_prediction_drift,
    score_prediction_drift_psi,
)


def test_evaluate_classification_bounds():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.3, 0.6, 0.4])
    m = evaluate_classification(y, p, threshold=0.5)
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"] <= 1
    assert "brier" in m


def test_expected_financial_loss_positive():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.3])
    loss = expected_financial_loss(y, p, threshold=0.5, avg_loan_amount=10_000)
    assert loss["expected_loss_total"] > 0
    assert loss["false_negatives"] >= 1
    assert loss["false_positives"] >= 1


def test_find_best_threshold_business():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0] * 5)
    p = np.clip(y + np.random.default_rng(0).normal(0, 0.2, size=len(y)), 0, 1)
    t = find_best_threshold(y, p, metric="business_loss")
    assert 0.1 <= t <= 0.85


def test_prediction_drift_stable_when_close():
    ref = [0.2, 0.25, 0.3, 0.22, 0.28] * 10
    cur = [0.21, 0.26, 0.29, 0.23, 0.27] * 10
    d = detect_prediction_drift(ref, cur)
    assert d["drift_detected"] is False
    assert "drift_score" in d


def test_psi_report_has_level():
    ref = list(np.random.default_rng(0).beta(2, 5, 200))
    cur = list(np.random.default_rng(1).beta(2, 5, 200))
    report = score_prediction_drift_psi(ref, cur)
    assert "psi" in report
    assert report["level"] in {"no_change", "small_change", "large_change"}


def test_feature_drift_api():
    out = detect_feature_drift([1.0, 2.0, 3.0], [1.01, 2.02, 2.99])
    assert "drift_detected" in out
    assert out["drift_detected"] is False
