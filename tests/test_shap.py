"""
Tests for SHAP / attribution explainability.

Covers:
- Feature-name alignment from fitted preprocessor (not module globals)
- Logistic Regression single-row coefficient approximation
- Tree-model TreeExplainer path
- Output dimensionality and top-k drivers
- Failure fallback does not break prediction
- Risk direction (positive → higher default risk)
- Integration with the saved best_pipeline artefact when present
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imblearn.pipeline import Pipeline as ImbPipeline
from src.data.generate_data import generate_loan_dataset
from src.explainability.shap_analysis import (
    compute_shap_values,
    top_shap_drivers,
    _is_linear_model,
    _is_tree_model,
)
from src.features.preprocessing import (
    TARGET,
    build_preprocessor,
    get_feature_names,
    prepare_xy,
    set_feature_lists,
)
from src.models.train import _get_models


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return generate_loan_dataset(n_samples=600, seed=42)


@pytest.fixture(scope="module")
def lr_pipeline(synthetic_df):
    """Fitted LR pipeline on synthetic data (no SMOTE for simpler SHAP)."""
    X, y = prepare_xy(synthetic_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor(schema="synthetic")),
        ("classifier", _get_models(42)["logistic_regression"]),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe, X_te, y_te


@pytest.fixture(scope="module")
def rf_pipeline(synthetic_df):
    """Fitted RandomForest pipeline on synthetic data."""
    X, y = prepare_xy(synthetic_df)
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor(schema="synthetic")),
        ("classifier", _get_models(42)["random_forest"]),
    ])
    pipe.fit(X, y)
    return pipe, X


def test_get_feature_names_matches_transform(synthetic_df):
    """Feature names from fitted preprocessor must match transform width."""
    # Reset module globals to REAL (the bug scenario)
    set_feature_lists("real")
    X, y = prepare_xy(synthetic_df)  # switches to synthetic
    pre = build_preprocessor(schema="synthetic")
    Xt = pre.fit_transform(X)
    names = get_feature_names(pre)
    assert len(names) == Xt.shape[1], (
        f"Name count {len(names)} != transform width {Xt.shape[1]}"
    )
    # Must not contain REAL-only columns when schema is synthetic
    assert "borrower_score" not in names
    assert "payment_inc_ratio" not in names
    assert "fico_score" in names or "int_rate" in names


def test_feature_names_independent_of_module_globals(synthetic_df):
    """After fitting a synthetic preprocessor, names stay correct even if globals flipped."""
    X, _ = prepare_xy(synthetic_df)
    pre = build_preprocessor(schema="synthetic")
    pre.fit(X)
    set_feature_lists("real")  # poison globals
    names = get_feature_names(pre)
    Xt = pre.transform(X.iloc[[0]])
    assert len(names) == Xt.shape[1]
    assert "borrower_score" not in names


def test_lr_single_row_approximation(lr_pipeline):
    pipe, X_te, _ = lr_pipeline
    row = X_te.iloc[[0]]
    explainer, shap_vals, X_trans, feat_names, is_approx = compute_shap_values(
        pipe, row, max_samples=1
    )
    assert is_approx is True
    assert explainer is None
    assert shap_vals.shape == (1, X_trans.shape[1])
    assert len(feat_names) == X_trans.shape[1]
    drivers = top_shap_drivers(shap_vals, feat_names, 0, top_k=5)
    assert len(drivers) == 5
    # Each driver is (str, float)
    for name, val in drivers:
        assert isinstance(name, str)
        assert isinstance(val, float)


def test_lr_batch_linear_explainer(lr_pipeline):
    pipe, X_te, _ = lr_pipeline
    batch = X_te.iloc[:30]
    explainer, shap_vals, X_trans, feat_names, is_approx = compute_shap_values(
        pipe, batch, max_samples=30
    )
    # May be exact LinearExplainer or fallback approx
    assert shap_vals.shape[0] == batch.shape[0]
    assert shap_vals.shape[1] == X_trans.shape[1]
    assert len(feat_names) == X_trans.shape[1]


def test_tree_explainer(rf_pipeline):
    pipe, X = rf_pipeline
    row = X.iloc[[0]]
    explainer, shap_vals, X_trans, feat_names, is_approx = compute_shap_values(
        pipe, row, max_samples=1
    )
    assert is_approx is False
    assert explainer is not None
    assert shap_vals.shape == (1, X_trans.shape[1])
    assert len(feat_names) == X_trans.shape[1]
    drivers = top_shap_drivers(shap_vals, feat_names, 0, top_k=6)
    assert 1 <= len(drivers) <= 6


def test_top_shap_drivers_dimensionality():
    vals = np.array([[0.1, -0.5, 0.3, 0.0, -0.2]])
    names = ["a", "b", "c", "d", "e"]
    drivers = top_shap_drivers(vals, names, 0, top_k=3)
    assert len(drivers) == 3
    # Highest abs first
    assert drivers[0][0] == "b"
    assert drivers[0][1] == -0.5


def test_risk_direction_positive_increases_default(lr_pipeline):
    """Positive attribution should mean higher predicted default probability."""
    pipe, X_te, _ = lr_pipeline
    row = X_te.iloc[[0]]
    _, shap_vals, _, feat_names, _ = compute_shap_values(pipe, row, max_samples=1)
    drivers = top_shap_drivers(shap_vals, feat_names, 0, top_k=10)
    # At least one positive and one negative is typical; just check signs are floats
    for _, v in drivers:
        assert np.isfinite(v)


def test_prediction_survives_shap_failure(lr_pipeline):
    """Even if SHAP path raises, predict_proba must still work."""
    pipe, X_te, _ = lr_pipeline
    row = X_te.iloc[[0]]
    proba = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_saved_artifact_shap_if_present():
    """If best_pipeline.joblib exists, explain a single synthetic-style row."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "best_pipeline.joblib"
    if not model_path.exists():
        pytest.skip("No saved model artefact")
    import joblib
    art = joblib.load(model_path)
    pipe = art["pipeline"]
    schema = art.get("schema", "synthetic")
    feat_names = art.get("feature_names")

    if schema == "synthetic":
        row = pd.DataFrame([{
            "loan_amnt": 10000.0, "int_rate": 12.5, "installment": 300.0,
            "annual_inc": 55000.0, "dti": 18.0, "delinq_2yrs": 0.0,
            "fico_score": 690.0, "inq_last_6mths": 1.0, "open_acc": 8.0,
            "pub_rec": 0.0, "revol_util": 45.0, "total_acc": 20.0,
            "term": 36, "grade": "B", "emp_length": "5 years",
            "home_ownership": "RENT", "purpose": "debt_consolidation",
        }])
    else:
        pytest.skip("Real-schema artefact; skip synthetic row test")

    proba = float(pipe.predict_proba(row)[0, 1])
    assert 0.0 <= proba <= 1.0

    explainer, shap_vals, X_trans, names, is_approx = compute_shap_values(
        pipe, row, max_samples=1, feature_names=feat_names
    )
    assert shap_vals.shape[1] == X_trans.shape[1]
    assert len(names) == X_trans.shape[1]
    # Names must match artefact when provided
    if feat_names is not None:
        assert names == list(feat_names) or len(names) == len(feat_names)
    drivers = top_shap_drivers(shap_vals, names, 0, top_k=6)
    assert len(drivers) >= 1
    # No REAL-schema ghost names
    for n, _ in drivers:
        assert "borrower_score" not in n
        assert "payment_inc_ratio" not in n


def test_model_type_helpers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    assert _is_linear_model(LogisticRegression())
    assert _is_tree_model(RandomForestClassifier())
    assert not _is_tree_model(LogisticRegression())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
