"""Application tests for the FastAPI prediction service."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_without_model(client):
    with patch("api.main.load_best_model", side_effect=FileNotFoundError):
        # Force reload path
        import api.main as m

        m._artifact = None
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")


def test_predict_validation_error(client):
    fake = {
        "pipeline": MagicMock(),
        "model_name": "logistic_regression",
        "threshold": 0.5,
        "schema": "real",
    }
    with patch("api.main._load_artifact", return_value=fake):
        resp = client.post(
            "/predict",
            json={"features": {"loan_amnt": 100}},  # missing required fields
        )
        assert resp.status_code == 422


def test_predict_happy_path(client):
    pipe = MagicMock()
    pipe.predict_proba.return_value = [[0.2, 0.75]]
    fake = {
        "pipeline": pipe,
        "model_name": "xgboost",
        "threshold": 0.5,
        "schema": "real",
    }
    features = {
        "loan_amnt": 12000,
        "term": 36,
        "annual_inc": 65000,
        "dti": 16.0,
        "payment_inc_ratio": 8.0,
        "revol_bal": 12000,
        "revol_util": 55.0,
        "purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE",
        "delinq_2yrs_zero": 1,
        "pub_rec_zero": 1,
        "open_acc": 10,
        "grade": 4.5,
        "emp_length": 5,
        "borrower_score": 0.55,
    }
    with patch("api.main._load_artifact", return_value=fake):
        resp = client.post("/predict", json={"schema": "real", "features": features})
        assert resp.status_code == 200
        body = resp.json()
        assert body["default_probability"] == pytest.approx(0.75, rel=1e-4)
        assert body["predicted_class"] == 1
        assert body["risk_tier"] == "High Risk"
        assert body["model_name"] == "xgboost"
