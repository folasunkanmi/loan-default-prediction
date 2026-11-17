"""Unit tests for prediction monitoring."""

from __future__ import annotations

import numpy as np

from src.monitoring.tracker import PredictionMonitor


def test_monitor_logs_and_distribution():
    mon = PredictionMonitor(max_records=100, persist=False)
    for p in [0.1, 0.2, 0.6, 0.8, 0.4]:
        mon.log(
            probability=p,
            predicted_class=int(p >= 0.5),
            threshold=0.5,
            model_name="test",
            schema="real",
            risk_tier="Medium Risk" if 0.3 <= p < 0.55 else ("Low Risk" if p < 0.3 else "High Risk"),
        )
    dist = mon.prediction_distribution()
    assert dist["count"] == 5
    assert 0.0 <= dist["mean_prob"] <= 1.0
    assert dist["class_rate"] is not None


def test_psi_insufficient_data():
    mon = PredictionMonitor(persist=False)
    result = mon.psi_vs_reference([0.1, 0.2, 0.3], last_n=10)
    assert result["status"] == "insufficient_data"


def test_psi_stable_when_similar():
    mon = PredictionMonitor(max_records=1000, persist=False)
    rng = np.random.default_rng(0)
    ref = rng.beta(2, 5, size=400).tolist()
    for p in ref:
        mon.log(
            probability=float(p),
            predicted_class=int(p >= 0.5),
            threshold=0.5,
            model_name="test",
            schema="real",
            risk_tier="Low Risk",
        )
    result = mon.psi_vs_reference(ref, last_n=400)
    assert result["psi"] is not None
    assert result["status"] in {"stable", "moderate_shift"}
