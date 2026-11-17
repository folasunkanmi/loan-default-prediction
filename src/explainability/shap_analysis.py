"""
SHAP explainability utilities.

Supports:
- Tree-based models (RandomForest, XGBoost, LightGBM) via TreeExplainer
- Linear models (Logistic Regression) via LinearExplainer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "lgbm", "lightgbm", "boosting", "tree"))


def compute_shap_values(
    pipeline,
    X_sample: pd.DataFrame,
    max_samples: int = 400,
) -> Tuple[Any, np.ndarray, np.ndarray, List[str]]:
    """
    Compute SHAP values for a fitted pipeline.

    Returns
    -------
    explainer, shap_values, X_transformed, feature_names
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X_sample)

    if X_transformed.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_transformed.shape[0], max_samples, replace=False)
        X_transformed = X_transformed[idx]

    # Recover feature names
    try:
        from src.features.preprocessing import get_feature_names
        feature_names = get_feature_names(preprocessor)
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
    else:
        # Linear / other models
        # Use a background sample for LinearExplainer
        background = shap.sample(X_transformed, min(100, X_transformed.shape[0]), random_state=42)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return explainer, np.asarray(shap_values), X_transformed, feature_names


def top_shap_drivers(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return the top-k features by absolute SHAP value for a single instance.
    Each item is (feature_name, shap_value).
    """
    if shap_values.ndim == 1:
        vals = shap_values
    else:
        vals = shap_values[instance_idx]
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [(feature_names[i], float(vals[i])) for i in order]


def plot_shap_summary(
    shap_values,
    X_transformed,
    feature_names: list,
    save: bool = True,
    filename: str = "shap_summary.png",
):
    """Global feature importance via SHAP summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP summary → {path}")
    plt.close()
