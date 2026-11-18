"""SHAP unwrap and explainability smoke tests."""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from src.data.generate_data import generate_loan_dataset
from src.explainability.shap_analysis import (
    compute_shap_values,
    explain_instance,
    top_shap_drivers,
    unwrap_pipeline,
)
from src.features.preprocessing import build_preprocessor, prepare_xy


def test_unwrap_plain_pipeline():
    df = generate_loan_dataset(n_samples=300, seed=0)
    X, y = prepare_xy(df)
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor("synthetic")),
        ("classifier", LogisticRegression(max_iter=500, random_state=0)),
    ])
    pipe.fit(X, y)
    assert unwrap_pipeline(pipe) is pipe


def test_shap_on_logistic_pipeline():
    df = generate_loan_dataset(n_samples=400, seed=1)
    X, y = prepare_xy(df)
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor("synthetic")),
        ("classifier", LogisticRegression(max_iter=800, random_state=1)),
    ])
    pipe.fit(X, y)
    row = X.iloc[[0]]
    _, shap_vals, Xt, names = compute_shap_values(pipe, row, max_samples=1)
    assert shap_vals is not None
    assert len(names) == Xt.shape[1]
    drivers = top_shap_drivers(shap_vals, names, top_k=3)
    assert len(drivers) == 3


def test_explain_instance_structure():
    df = generate_loan_dataset(n_samples=400, seed=2)
    X, y = prepare_xy(df)
    pipe = ImbPipeline([
        ("preprocessor", build_preprocessor("synthetic")),
        ("classifier", LogisticRegression(max_iter=800, random_state=2)),
    ])
    pipe.fit(X, y)
    out = explain_instance(pipe, X.iloc[[5]], top_k=4)
    assert len(out) == 4
    assert {"feature", "shap_value", "direction"} <= set(out[0].keys())
