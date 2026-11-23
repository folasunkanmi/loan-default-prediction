"""Train, select, and evaluate loan-default models with a held-out test set."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data
from src.evaluation.metrics import plot_confusion_matrix, plot_roc_pr_curves
from src.features.preprocessing import (
    build_preprocessor,
    detect_schema,
    get_feature_names,
    prepare_xy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42


def _get_models(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate_model(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(y_true, y_proba) -> float:
    """Choose the threshold that maximises positive-class F1."""
    thresholds = np.linspace(0.15, 0.75, 61)
    scores = [
        f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def _make_pipeline(
    model_name: str,
    classifier,
    schema: str,
    feature_columns: list[str],
    apply_smote: bool,
    random_state: int,
) -> ImbPipeline:
    steps = [
        (
            "preprocessor",
            build_preprocessor(schema=schema, feature_columns=feature_columns),
        )
    ]
    if model_name == "logistic_regression" and apply_smote:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=5)))
    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


def train_all_models(
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train candidates, select/tune using validation, and evaluate test once."""
    if not 0 < test_size < 1 or not 0 < val_size < 1 - test_size:
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")

    df = load_raw_data()
    data_provenance = dict(df.attrs.get("provenance", {}))
    schema = detect_schema(df)
    X, y = prepare_xy(df)
    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    X_remainder, X_test, y_remainder, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_remainder,
        y_remainder,
        test_size=relative_val_size,
        stratify=y_remainder,
        random_state=random_state,
    )
    split_sizes = {
        "train": len(X_train),
        "validation": len(X_val),
        "test": len(X_test),
    }
    if sum(split_sizes.values()) != len(df):
        raise RuntimeError("Train/validation/test splits do not cover the dataset exactly.")

    # Oversampling only helps when the training partition is imbalanced.
    class_counts = y_train.value_counts()
    apply_smote = bool(class_counts.min() / class_counts.max() < 0.8)
    models = _get_models(random_state)
    validation_results: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}

    for model_name, classifier in models.items():
        pipeline = _make_pipeline(
            model_name,
            classifier,
            schema,
            list(X.columns),
            apply_smote,
            random_state,
        )
        pipeline.fit(X_train, y_train)
        validation_proba = pipeline.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, validation_proba)
        metrics = evaluate_model(y_val, validation_proba, threshold)
        metrics["best_threshold"] = threshold
        validation_results[model_name] = metrics
        thresholds[model_name] = threshold
        print(
            f"[VAL] {model_name:22s} ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1']:.4f} | "
            f"threshold={threshold:.3f}"
        )

    selected_model = max(
        validation_results,
        key=lambda name: validation_results[name]["pr_auc"],
    )
    selected_threshold = thresholds[selected_model]

    # Refit the validation-selected model on train + validation, without touching test.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_pipeline = _make_pipeline(
        selected_model,
        _get_models(random_state)[selected_model],
        schema,
        list(X.columns),
        apply_smote,
        random_state,
    )
    final_pipeline.fit(X_trainval, y_trainval)

    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_proba, selected_threshold)
    test_metrics["best_threshold"] = selected_threshold
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        (test_proba >= selected_threshold).astype(int),
        labels=[0, 1],
    ).ravel()
    test_confusion = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_pr_curves(
        y_test,
        {selected_model: test_proba},
        filename="roc_pr_curves_test.png",
    )
    plot_confusion_matrix(
        y_test,
        (test_proba >= selected_threshold).astype(int),
        title=f"Test Confusion Matrix — {selected_model}",
    )

    from src.config.settings import VERSION

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    default_rate = float(y.mean())
    provenance = {
        **data_provenance,
        "schema": schema,
        "n_samples": len(df),
        "default_rate": round(default_rate, 6),
        "class_counts": {
            "paid_off": int((y == 0).sum()),
            "default": int((y == 1).sum()),
        },
        "selected_model": selected_model,
        "selection_metric": "validation PR-AUC",
        "decision_threshold": selected_threshold,
        "threshold_selection": "F1 maximisation on validation set only",
        "split_sizes": split_sizes,
        "random_seed": random_state,
        "smote_applied": apply_smote,
        "trained_at": trained_at,
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "platform": platform.platform(),
        "software_version": VERSION,
        "note": (
            "The real dataset is a balanced curated sample (50% default). "
            "Its metrics do not estimate real-world Lending Club lending performance."
            if schema == "real"
            else "Metrics are from synthetic data, not real Lending Club performance."
        ),
    }
    artifact = {
        "pipeline": final_pipeline,
        "model_name": selected_model,
        "threshold": selected_threshold,
        "schema": schema,
        "feature_names": get_feature_names(
            final_pipeline.named_steps["preprocessor"]
        ),
        "raw_feature_names": list(X.columns),
        "val_results": validation_results,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_confusion,
        "provenance": provenance,
        "sklearn_version": sklearn.__version__,
        "software_version": VERSION,
        "trained_at": trained_at,
    }
    model_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(artifact, model_path)

    report = {
        "provenance": provenance,
        "evaluation_protocol": {
            "split_strategy": "stratified train/validation/test",
            "model_selection": "validation PR-AUC only",
            "threshold_selection": "validation F1 only",
            "final_test_evaluation": "selected model evaluated once on untouched test data",
        },
        "validation": validation_results,
        "test": {
            "model": selected_model,
            "metrics": test_metrics,
            "confusion_matrix": test_confusion,
        },
        "selected_model": selected_model,
        "decision_threshold": selected_threshold,
        "artifacts": {
            "model": "models/best_pipeline.joblib",
            "roc_pr_plot": "reports/roc_pr_curves_test.png",
            "confusion_matrix_plot": "reports/confusion_matrix.png",
        },
    }
    with (REPORTS_DIR / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")

    print(
        f"[TEST] {selected_model} ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"PR-AUC: {test_metrics['pr_auc']:.4f} | F1: {test_metrics['f1']:.4f}"
    )
    print(f"Saved model artifact → {model_path}")
    return {
        "validation": validation_results,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_confusion,
        "best_model": selected_model,
        "best_pipeline": final_pipeline,
        "threshold": selected_threshold,
        "schema": schema,
        "provenance": provenance,
    }


def load_best_model():
    path = MODELS_DIR / "best_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.models.train` first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Unexpected model artifact format at {path}; retrain the model.")
    if "threshold" not in artifact or not 0 < float(artifact["threshold"]) < 1:
        raise ValueError(f"Model artifact at {path} has no valid decision threshold.")
    return artifact


if __name__ == "__main__":
    train_all_models()
