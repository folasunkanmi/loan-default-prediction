"""SHAP explainability utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def compute_shap_values(pipeline, X_sample: pd.DataFrame, max_samples: int = 500):
    """Compute SHAP values for a fitted pipeline that ends with a tree-based model."""
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_sample)

    if X_transformed.shape[0] > max_samples:
        idx = np.random.choice(X_transformed.shape[0], max_samples, replace=False)
        X_transformed = X_transformed[idx]

    model = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values, X_transformed


def plot_shap_summary(shap_values, X_transformed, feature_names: list, save: bool = True, filename: str = "shap_summary.png"):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP summary → {path}")
    plt.close()
