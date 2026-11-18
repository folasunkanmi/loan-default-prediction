"""
Train, calibrate, and compare classifiers for loan default prediction.

Protocol
--------
- Stratified Train / Validation / Test
- Model selection + threshold tuning on VALIDATION only
- Probability calibration fitted on VALIDATION
- Test set evaluated once (final report)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data, save_processed_data
from src.evaluation.metrics import (
    evaluate_classification,
    expected_financial_loss,
    find_best_threshold,
    plot_calibration_curve,
    plot_roc_pr_curves,
)
from src.features.preprocessing import (
    TARGET,
    build_preprocessor,
    detect_schema,
    get_feature_names,
    prepare_xy,
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


def _build_pipe(name: str, clf, schema: str, random_state: int) -> ImbPipeline:
    steps = [("preprocessor", build_preprocessor(schema=schema))]
    if name == "logistic_regression":
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=5)))
    steps.append(("classifier", clf))
    return ImbPipeline(steps=steps)


def train_all_models(
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = RANDOM_SEED,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> Dict[str, Any]:
    print("Loading data...")
    df = load_raw_data()
    schema = detect_schema(df)
    print(f"Detected schema: {schema} | Shape: {df.shape} | Default rate: {df[TARGET].mean():.3f}")

    X, y = prepare_xy(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=random_state
    )

    print(f"Split sizes → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

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
    val_probas: Dict[str, np.ndarray] = {}

    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        pipe = _build_pipe(name, clf, schema, random_state)
        pipe.fit(X_train, y_train)

        if calibrate:
            # Calibrate on validation fold (prefit base estimator)
            calibrator = CalibratedClassifierCV(
                pipe, method=calibration_method, cv="prefit"
            )
            calibrator.fit(X_val, y_val)
            model_for_pred = calibrator
        else:
            model_for_pred = pipe

        y_val_proba = model_for_pred.predict_proba(X_val)[:, 1]
        thr = find_best_threshold(y_val, y_val_proba, metric="f1")
        metrics = evaluate_classification(y_val, y_val_proba, thr)
        biz = expected_financial_loss(y_val, y_val_proba, thr)
        metrics.update({
            "best_threshold": thr,
            "expected_loss_per_loan": biz["expected_loss_per_loan"],
            "calibrated": calibrate,
        })

        val_results[name] = metrics
        fitted_pipelines[name] = model_for_pred
        thresholds[name] = thr
        val_probas[name] = y_val_proba

        print(
            f"  [VAL] ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"P: {metrics['precision']:.3f} R: {metrics['recall']:.3f} | "
            f"F1: {metrics['f1']:.4f} | Brier: {metrics['brier']:.4f} | "
            f"Loss/loan: {metrics['expected_loss_per_loan']:.1f} | thr={thr:.3f}"
        )

    # Calibration plot on validation
    try:
        plot_calibration_curve(y_val, val_probas, filename="calibration_curve_val.png")
        plot_roc_pr_curves(y_val, val_probas, filename="roc_pr_curves_val.png")
    except Exception as exc:
        print(f"Plot warning: {exc}")

    best_name = max(val_results, key=lambda k: val_results[k]["pr_auc"])
    best_thr = thresholds[best_name]
    print(f"\nBest model on validation: {best_name} (PR-AUC={val_results[best_name]['pr_auc']:.4f})")

    # Re-fit on Train+Val, then calibrate on a small holdout from trainval if possible
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    best_clf = _get_models(random_state)[best_name]
    base_pipe = _build_pipe(best_name, best_clf, schema, random_state)

    if calibrate and len(X_trainval) > 200:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=random_state
        )
        base_pipe.fit(X_fit, y_fit)
        final_model = CalibratedClassifierCV(base_pipe, method=calibration_method, cv="prefit")
        final_model.fit(X_cal, y_cal)
    else:
        base_pipe.fit(X_trainval, y_trainval)
        final_model = base_pipe

    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_classification(y_test, y_test_proba, best_thr)
    test_biz = expected_financial_loss(y_test, y_test_proba, best_thr)
    test_metrics.update({
        "best_threshold": best_thr,
        "expected_loss_per_loan": test_biz["expected_loss_per_loan"],
        "expected_loss_total": test_biz["expected_loss_total"],
        "false_negatives": test_biz["false_negatives"],
        "false_positives": test_biz["false_positives"],
        "calibrated": calibrate,
    })
    print(
        f"\n[TEST] {best_name} → ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"PR-AUC: {test_metrics['pr_auc']:.4f} | "
        f"P: {test_metrics['precision']:.3f} R: {test_metrics['recall']:.3f} | "
        f"F1: {test_metrics['f1']:.4f} | "
        f"Expected loss/loan: {test_metrics['expected_loss_per_loan']:.1f}"
    )

    try:
        plot_calibration_curve(
            y_test, {best_name: y_test_proba}, filename="calibration_curve_test.png"
        )
        plot_roc_pr_curves(
            y_test, {best_name: y_test_proba}, filename="roc_pr_curves_test.png"
        )
    except Exception as exc:
        print(f"Test plot warning: {exc}")

    # Feature reference stats for drift monitoring
    ref_stats = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            ref_stats[col] = {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std() or 1.0),
                "p10": float(X_train[col].quantile(0.10)),
                "p50": float(X_train[col].quantile(0.50)),
                "p90": float(X_train[col].quantile(0.90)),
                "type": "numeric",
            }
        else:
            vc = X_train[col].astype(str).value_counts(normalize=True)
            ref_stats[col] = {
                "frequencies": vc.to_dict(),
                "type": "categorical",
            }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve feature names for SHAP when possible
    try:
        if hasattr(final_model, "estimator"):
            pre = final_model.estimator.named_steps["preprocessor"]
        else:
            pre = final_model.named_steps["preprocessor"]
        feat_names = get_feature_names(pre)
    except Exception:
        feat_names = list(X.columns)

    artifact = {
        "pipeline": final_model,
        "model_name": best_name,
        "threshold": best_thr,
        "schema": schema,
        "feature_columns": list(X.columns),
        "feature_names": feat_names,
        "val_results": val_results,
        "test_metrics": test_metrics,
        "calibrated": calibrate,
        "calibration_method": calibration_method if calibrate else None,
        "reference_stats": ref_stats,
        "reference_probas": y_test_proba[:500].tolist(),
    }
    model_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(artifact, model_path)
    print(f"Saved → {model_path}")

    report = {
        "validation": val_results,
        "test": {best_name: test_metrics},
        "selected_model": best_name,
        "schema": schema,
        "calibrated": calibrate,
    }
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        "val_results": val_results,
        "test_metrics": test_metrics,
        "best_model": best_name,
        "best_pipeline": final_model,
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
