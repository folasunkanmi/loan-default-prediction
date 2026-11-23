"""
SHAP explainability utilities.

Supports:
- Tree-based models (RandomForest, XGBoost, LightGBM) via TreeExplainer
- Linear models (Logistic Regression) via LinearExplainer + coefficient fallback
- KernelExplainer as last-resort fallback

For single-row Logistic Regression explanations the coefficient × standardised
feature value approximation is used when a background dataset is unavailable.
This is labelled as an approximation and is not exact SHAP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "lgbm", "lightgbm", "boosting", "tree"))


def _is_linear_model(model) -> bool:
    return isinstance(model, LogisticRegression) or "logistic" in type(model).__name__.lower()


def _to_dense(X):
    """Convert sparse matrix to dense ndarray if needed."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _resolve_feature_names(
    preprocessor,
    X_transformed: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> List[str]:
    """Prefer caller-supplied names, then fitted preprocessor, then generic."""
    n_feats = X_transformed.shape[1]

    if feature_names is not None and len(feature_names) == n_feats:
        return list(feature_names)

    try:
        from src.features.preprocessing import get_feature_names
        names = get_feature_names(preprocessor)
        if len(names) == n_feats:
            return names
    except Exception as e:
        logger.debug("get_feature_names failed: %s", e)

    return [f"f{i}" for i in range(n_feats)]


def compute_shap_values(
    pipeline,
    X_sample: pd.DataFrame,
    max_samples: int = 400,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Any, np.ndarray, np.ndarray, List[str], bool]:
    """
    Compute SHAP values for a fitted pipeline.

    Parameters
    ----------
    pipeline : fitted sklearn / imblearn Pipeline with 'preprocessor' and 'classifier'
    X_sample : raw input features (before preprocessing)
    max_samples : subsample limit for large batches
    feature_names : optional list of transformed feature names (e.g. from model artifact).
                    Preferred over deriving names from module-level globals.

    Returns
    -------
    explainer : SHAP explainer instance or None (when approximation used)
    shap_values : ndarray of shape (n_samples, n_features)
    X_transformed : dense transformed feature matrix
    feature_names : list of length n_features
    is_approximation : True when coefficient×value heuristic was used instead of true SHAP
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    X_transformed = _to_dense(preprocessor.transform(X_sample))

    if X_transformed.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_transformed.shape[0], max_samples, replace=False)
        X_transformed = X_transformed[idx]

    feat_names = _resolve_feature_names(preprocessor, X_transformed, feature_names)

    # --- Tree models ---
    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        # Binary classification: shap may return list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive (default) class
        shap_values = np.asarray(shap_values)
        # Newer SHAP versions may return (n, n_features, n_classes)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        return explainer, shap_values, X_transformed, feat_names, False

    # --- Linear models ---
    if _is_linear_model(model):
        n_rows = X_transformed.shape[0]
        # Single-row case: LinearExplainer needs a background; use coefficient approx
        if n_rows == 1:
            coefs = np.asarray(model.coef_).ravel()
            if coefs.shape[0] != X_transformed.shape[1]:
                raise ValueError(
                    f"Coefficient length {coefs.shape[0]} != transformed features {X_transformed.shape[1]}"
                )
            # Data is StandardScaler-centred → coef * x is a reasonable local attribution
            approx = (coefs * X_transformed[0]).reshape(1, -1)
            logger.info(
                "Using coefficient×value approximation for single-row LogisticRegression "
                "(not exact SHAP)."
            )
            return None, approx, X_transformed, feat_names, True

        try:
            n_bg = min(50, n_rows)
            background = shap.sample(X_transformed, n_bg, random_state=42)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = np.asarray(shap_values)
            return explainer, shap_values, X_transformed, feat_names, False
        except Exception as e:
            logger.warning("LinearExplainer failed (%s); falling back to coefficient approximation", e)
            coefs = np.asarray(model.coef_).ravel()
            approx = coefs * X_transformed
            return None, np.asarray(approx), X_transformed, feat_names, True

    # --- Last resort: KernelExplainer (slow but general) ---
    try:
        n_bg = min(20, X_transformed.shape[0])
        background = shap.sample(X_transformed, n_bg, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_transformed, nsamples=50)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)
        return explainer, shap_values, X_transformed, feat_names, False
    except Exception as e:
        raise RuntimeError(f"Could not compute SHAP values: {e}") from e


def top_shap_drivers(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Return the top-k features by absolute SHAP value for a single instance.
    Each item is (feature_name, shap_value).
    Positive values increase predicted default probability; negative decrease it.
    """
    if shap_values is None or len(feature_names) == 0:
        return []

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        vals = shap_values
    else:
        vals = shap_values[instance_idx]

    # Safety: align lengths
    n = min(len(vals), len(feature_names))
    vals = vals[:n]
    names = feature_names[:n]

    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [(names[i], float(vals[i])) for i in order]


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
