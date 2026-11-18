"""
SHAP explainability utilities.

Handles:
- Plain sklearn/imblearn pipelines
- CalibratedClassifierCV wrappers (uses underlying estimator)
- Tree models via TreeExplainer; linear via LinearExplainer; fallback KernelExplainer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def unwrap_pipeline(model: Any) -> Any:
    """
    Return the base imblearn/sklearn pipeline that has named_steps
    preprocessor + classifier, even when wrapped by CalibratedClassifierCV.
    """
    # CalibratedClassifierCV (cv='prefit') stores the fitted estimator here
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        # sklearn >= 1.4 uses .estimator; older used .base_estimator
        est = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
        if est is not None:
            return est
    if hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        return model.estimator
    if hasattr(model, "base_estimator") and hasattr(model.base_estimator, "named_steps"):
        return model.base_estimator
    return model


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("forest", "xgb", "lgbm", "lightgbm", "boosting", "tree"))


def compute_shap_values(
    pipeline,
    X_sample: pd.DataFrame,
    max_samples: int = 400,
) -> Tuple[Any, np.ndarray, np.ndarray, List[str]]:
    """
    Compute SHAP values for a fitted pipeline or calibrated wrapper.

    Returns
    -------
    explainer, shap_values, X_transformed, feature_names
    """
    base = unwrap_pipeline(pipeline)
    if not hasattr(base, "named_steps"):
        raise ValueError(
            "Cannot extract preprocessor/classifier from model. "
            f"Got type={type(pipeline).__name__}"
        )

    preprocessor = base.named_steps["preprocessor"]
    model = base.named_steps["classifier"]
    X_transformed = preprocessor.transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed, dtype=float)

    if X_transformed.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_transformed.shape[0], max_samples, replace=False)
        X_transformed = X_transformed[idx]

    try:
        from src.features.preprocessing import get_feature_names

        feature_names = get_feature_names(preprocessor)
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    if len(feature_names) != X_transformed.shape[1]:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        n_bg = min(100, X_transformed.shape[0])
        background = shap.sample(X_transformed, n_bg, random_state=42)
        try:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_transformed)
        except Exception:
            # Slow but works for arbitrary models on small batches
            explainer = shap.KernelExplainer(
                model.predict_proba, background[: min(50, n_bg)]
            )
            raw = explainer.shap_values(X_transformed[: min(20, len(X_transformed))], nsamples=50)
            shap_values = raw[1] if isinstance(raw, list) else raw

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return explainer, np.asarray(shap_values, dtype=float), X_transformed, feature_names


def top_shap_drivers(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    if shap_values.ndim == 1:
        vals = shap_values
    else:
        vals = shap_values[min(instance_idx, shap_values.shape[0] - 1)]
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [(feature_names[i], float(vals[i])) for i in order if i < len(feature_names)]


def explain_instance(
    pipeline,
    X_row: pd.DataFrame,
    top_k: int = 6,
) -> List[dict]:
    """Convenience: return list of {feature, shap_value, direction}."""
    _, shap_vals, _, feat_names = compute_shap_values(pipeline, X_row, max_samples=1)
    drivers = top_shap_drivers(shap_vals, feat_names, instance_idx=0, top_k=top_k)
    out = []
    for name, val in drivers:
        out.append({
            "feature": name,
            "shap_value": round(val, 6),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        })
    return out


def plot_shap_summary(
    shap_values,
    X_transformed,
    feature_names: list,
    save: bool = True,
    filename: str = "shap_summary.png",
):
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
