"""
Train and compare Logistic Regression, Random Forest, XGBoost and LightGBM.

Proper protocol:
- Train / Validation / Test split (stratified)
- Model selection and threshold tuning performed on VALIDATION set only
- Test set used once for final unbiased evaluation
"""

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
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data, save_processed_data
from src.features.preprocessing import (
    build_preprocessor,
    detect_schema,
    get_feature_names,
    prepare_xy,
    TARGET,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42


def _get_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
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
    """Choose threshold that maximises F1 on the positive class (validation only)."""
    thresholds = np.linspace(0.15, 0.75, 61)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def train_all_models(
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Full training pipeline with proper isolation of the test set.

    Protocol
    --------
    1. Load data and detect schema (real vs synthetic).
    2. Split into Train / Val / Test (stratified).
    3. For each model:
         - Fit on Train (SMOTE inside pipeline for LR)
         - Evaluate + tune threshold on Validation
    4. Select best model by validation PR-AUC.
    5. Re-fit best model on Train+Val.
    6. Final evaluation on held-out Test (once).
    """
    print("Loading data...")
    df = load_raw_data()
    schema = detect_schema(df)
    print(f"Detected schema: {schema} | Shape: {df.shape} | Default rate: {df[TARGET].mean():.3f}")

    X, y = prepare_xy(df)

    # First split: hold out the final test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Second split: train vs validation from the remainder
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=random_state
    )

    print(f"Split sizes → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Persist splits for reproducibility / later analysis
    for name, Xx, yy in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        tmp = Xx.copy()
        tmp[TARGET] = yy.values
        save_processed_data(tmp, f"{name}.parquet")

    models = _get_models(random_state)
    val_results: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Any] = {}
    thresholds: Dict[str, float] = {}

    for name, clf in models.items():
        print(f"\nTraining {name} ...")

        if name == "logistic_regression":
            pipe = ImbPipeline(
                steps=[
                    ("preprocessor", build_preprocessor(schema=schema)),
                    ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
                    ("classifier", clf),
                ]
            )
        else:
            pipe = ImbPipeline(
                steps=[
                    ("preprocessor", build_preprocessor(schema=schema)),
                    ("classifier", clf),
                ]
            )

        pipe.fit(X_train, y_train)
        y_val_proba = pipe.predict_proba(X_val)[:, 1]
        thr = find_best_threshold(y_val, y_val_proba)
        metrics = evaluate_model(y_val, y_val_proba, thr)
        metrics["best_threshold"] = thr

        val_results[name] = metrics
        fitted_pipelines[name] = pipe
        thresholds[name] = thr

        print(
            f"  [VAL] ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f} | thr={thr:.3f}"
        )

    # Select best model by validation PR-AUC
    best_name = max(val_results, key=lambda k: val_results[k]["pr_auc"])
    best_thr = thresholds[best_name]
    print(f"\nBest model on validation: {best_name} (PR-AUC={val_results[best_name]['pr_auc']:.4f})")

    # Re-fit best model on Train + Validation
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    best_clf = _get_models(random_state)[best_name]
    if best_name == "logistic_regression":
        final_pipe = ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(schema=schema)),
                ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
                ("classifier", best_clf),
            ]
        )
    else:
        final_pipe = ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(schema=schema)),
                ("classifier", best_clf),
            ]
        )
    final_pipe.fit(X_trainval, y_trainval)

    # Final evaluation on held-out TEST set (once)
    y_test_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_proba, best_thr)
    test_metrics["best_threshold"] = best_thr
    print(
        f"\n[TEST] {best_name} → ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"PR-AUC: {test_metrics['pr_auc']:.4f} | F1: {test_metrics['f1']:.4f}"
    )

    # Persist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": final_pipe,
        "model_name": best_name,
        "threshold": best_thr,
        "schema": schema,
        "feature_names": get_feature_names(final_pipe.named_steps["preprocessor"]),
        "val_results": val_results,
        "test_metrics": test_metrics,
    }
    model_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(artifact, model_path)
    print(f"Saved → {model_path}")

    report = {
        "validation": val_results,
        "test": {best_name: test_metrics},
        "selected_model": best_name,
    }
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        "val_results": val_results,
        "test_metrics": test_metrics,
        "best_model": best_name,
        "best_pipeline": final_pipe,
        "threshold": best_thr,
        "schema": schema,
    }


def load_best_model():
    path = MODELS_DIR / "best_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.models.train` first."
        )
    return joblib.load(path)


if __name__ == "__main__":
    train_all_models()
