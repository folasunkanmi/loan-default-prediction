"""Unit tests for prediction monitoring."""

from __future__ import annotations

import numpy as np

from src.monitoring import (
    PredictionMonitor,
    detect_feature_drift,
    detect_prediction_drift,
    score_prediction_drift_psi,
)


def test_imports_and_basic_drift():
    # Package-level imports must work
    mon = PredictionMonitor(max_records=100, persist=False)
    for p in [0.1, 0.2, 0.6, 0.8, 0.4]:
        mon.log_prediction(
            probability=p,
            predicted_class=int(p >= 0.5),
            threshold=0.5,
            model_name="test",
            schema="real",
            risk_tier="Medium Risk",
        )
    dist = mon.prediction_distribution()
    assert dist["count"] == 5
    assert 0.0 <= dist["mean_prob"] <= 1.0

    ref = [0.2, 0.25, 0.3, 0.22, 0.28]
    cur = [0.21, 0.26, 0.29, 0.23, 0.27]
    d = detect_prediction_drift(ref, cur)
    assert "drift_score" in d
    assert d["drift_detected"] is False

    psi = score_prediction_drift_psi(ref * 20, cur * 20)
    assert "psi" in psi
    assert psi["level"] in {"no_change", "small_change", "large_change"}

    fd = detect_feature_drift([1.0, 2.0, 3.0], [1.05, 2.01, 2.98])
    assert "drift_detected" in fd
    assert fd["drift_detected"] is False

    # Compatibility: log alias
    mon.log(0.5, 1, 0.5, "t", "real", "High Risk")
    assert len(mon.recent_probas()) == 6
