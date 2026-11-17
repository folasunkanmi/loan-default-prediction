"""Train and compare Logistic Regression, Random Forest, XGBoost and LightGBM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data, save_processed_data
from src.features.preprocessing import (
    build_preprocessor, get_feature_names, prepare_xy, TARGET,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _get_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state, solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=15,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.8, scale_pos_weight=1.0, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05, num_leaves=63,
            subsample=0.85, colsample_bytree=0.8, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbose=-1,
        ),
    }


def evaluate_model(y_true, y_proba, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(y_true, y_proba) -> float:
    thresholds = np.linspace(0.15, 0.75, 61)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def train_all_models(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    print("Loading data...")
    df = load_raw_data()
    print(f"Shape: {df.shape} | Default rate: {df[TARGET].mean():.3f}")

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    save_processed_data(train_df, "train.parquet")
    save_processed_data(test_df, "test.parquet")

    models = _get_models(random_state)
    results = {}
    best_name, best_pr_auc, best_pipeline, best_threshold = None, -1.0, None, 0.5

    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        if name == "logistic_regression":
            pipe = ImbPipeline([
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
                ("classifier", clf),
            ])
        else:
            pipe = ImbPipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ])

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        threshold = find_best_threshold(y_test, y_proba)
        metrics = evaluate_model(y_test, y_proba, threshold)
        metrics["best_threshold"] = threshold
        results[name] = metrics
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f}")

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_name = name
            best_pipeline = pipe
            best_threshold = threshold

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "threshold": best_threshold,
        "feature_names": get_feature_names(best_pipeline.named_steps["preprocessor"]),
    }, model_path)
    print(f"\nBest model: {best_name} (PR-AUC={best_pr_auc:.4f})")
    print(f"Saved → {model_path}")

    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return {"results": results, "best_model": best_name, "best_pipeline": best_pipeline, "threshold": best_threshold}


def load_best_model():
    path = MODELS_DIR / "best_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError("No trained model found. Run `python -m src.models.train` first.")
    return joblib.load(path)


if __name__ == "__main__":
    train_all_models()
