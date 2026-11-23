"""
Smoke and integration tests for the loan default prediction pipeline.

Covers:
- Synthetic data generation
- Schema detection & preprocessing fit/transform
- End-to-end training on synthetic data (all four models path)
- Prediction on a new applicant row
- No target leakage into feature lists
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_data import generate_loan_dataset
from src.features.preprocessing import (
    TARGET,
    build_preprocessor,
    detect_schema,
    get_feature_names,
    prepare_xy,
    set_feature_lists,
)
from src.models.train import _get_models, evaluate_model, find_best_threshold
from imblearn.pipeline import Pipeline as ImbPipeline


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return generate_loan_dataset(n_samples=800, seed=42)


def test_generate_data_shape_and_target(synthetic_df):
    assert len(synthetic_df) == 800
    assert TARGET in synthetic_df.columns
    assert set(synthetic_df[TARGET].unique()).issubset({0, 1})


def test_generate_data_default_rate(synthetic_df):
    rate = synthetic_df[TARGET].mean()
    assert 0.10 < rate < 0.35


def test_schema_detection_synthetic(synthetic_df):
    assert detect_schema(synthetic_df) == "synthetic"


def test_preprocessor_fit_transform(synthetic_df):
    X, y = prepare_xy(synthetic_df)
    assert X.shape[0] == len(y)
    assert TARGET not in X.columns

    pre = build_preprocessor(schema="synthetic")
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] > 0
    names = get_feature_names(pre)
    assert len(names) == Xt.shape[1]


def test_no_target_leakage():
    from src.features import preprocessing as pp
    set_feature_lists("synthetic")
    assert TARGET not in pp.NUMERIC_FEATURES
    assert TARGET not in pp.CATEGORICAL_FEATURES
    set_feature_lists("real")
    assert TARGET not in pp.NUMERIC_FEATURES
    assert TARGET not in pp.CATEGORICAL_FEATURES


def test_end_to_end_synthetic_training(synthetic_df):
    """Train one model on synthetic data and produce predictions."""
    X, y = prepare_xy(synthetic_df)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=1 / 3, stratify=y_trainval, random_state=42
    )

    clf = _get_models(42)["logistic_regression"]
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor(schema="synthetic")),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_proba)

    pipe.fit(X_trainval, y_trainval)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    assert test_proba.shape == (len(X_test),)
    assert np.all((test_proba >= 0) & (test_proba <= 1))

    metrics = evaluate_model(y_test, test_proba, threshold)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_prediction_on_new_applicant(synthetic_df):
    """Simulate a single new applicant after training."""
    X, y = prepare_xy(synthetic_df)
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor(schema="synthetic")),
        ("classifier", _get_models(42)["random_forest"]),
    ])
    pipe.fit(X, y)

    # One-row DataFrame with the same columns
    new_row = X.iloc[[0]].copy()
    proba = pipe.predict_proba(new_row)[0, 1]
    pred = int(proba >= 0.5)
    assert 0.0 <= proba <= 1.0
    assert pred in (0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
