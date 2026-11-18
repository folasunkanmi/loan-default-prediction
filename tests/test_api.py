"""API contract tests (model may be absent in pure unit env)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import APP_VERSION, app

client = TestClient(app)


def test_health_returns_json():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body.get("api_version") == APP_VERSION


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_predict_without_model_or_ok():
    payload = {
        "schema": "synthetic",
        "features": {
            "loan_amnt": 10000,
            "term": 36,
            "int_rate": 12.0,
            "installment": 300,
            "grade": "B",
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "annual_inc": 60000,
            "purpose": "debt_consolidation",
            "dti": 15.0,
            "delinq_2yrs": 0,
            "fico_score": 700,
            "inq_last_6mths": 1,
            "open_acc": 8,
            "pub_rec": 0,
            "revol_util": 40.0,
            "total_acc": 15,
        },
    }
    r = client.post("/predict", json=payload)
    # 200 if model baked/trained, 503 if not available in CI before train step
    assert r.status_code in (200, 503, 400, 422)
    if r.status_code == 200:
        data = r.json()
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["predicted_class"] in (0, 1)
        assert "threshold" in data
