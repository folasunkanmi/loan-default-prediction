"""Evaluation helpers: classification metrics, calibration, business loss, plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# Default cost assumptions (can be overridden)
DEFAULT_COST_FN = 1.0   # cost of missing a default (false negative)
DEFAULT_COST_FP = 0.2   # cost of rejecting a good borrower (false positive)
DEFAULT_AVG_LOAN = 12_000.0


def evaluate_classification(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
    }


def expected_financial_loss(
    y_true,
    y_proba,
    threshold: float = 0.5,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    avg_loan_amount: float = DEFAULT_AVG_LOAN,
) -> Dict[str, float]:
    """
    Business-oriented loss under a simple cost matrix.

    FN (approve a defaulter): lose `cost_fn * avg_loan_amount`
    FP (reject a good borrower): opportunity cost `cost_fp * avg_loan_amount`

    Returns per-loan expected loss and aggregate counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n = max(len(y_true), 1)

    total_loss = fn * cost_fn * avg_loan_amount + fp * cost_fp * avg_loan_amount
    return {
        "expected_loss_total": float(total_loss),
        "expected_loss_per_loan": float(total_loss / n),
        "false_negatives": fn,
        "false_positives": fp,
        "true_positives": tp,
        "true_negatives": tn,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "avg_loan_amount": avg_loan_amount,
    }


def find_best_threshold(
    y_true,
    y_proba,
    metric: str = "f1",
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """
    Tune threshold on validation only.
    metric: 'f1' | 'recall' | 'business_loss' (minimises expected loss)
    """
    thresholds = np.linspace(0.10, 0.85, 76)
    best_t = 0.5
    if metric == "business_loss":
        best_score = float("inf")
        for t in thresholds:
            loss = expected_financial_loss(y_true, y_proba, t, cost_fn, cost_fp)[
                "expected_loss_per_loan"
            ]
            if loss < best_score:
                best_score = loss
                best_t = float(t)
        return best_t

    best_score = -1.0
    for t in thresholds:
        m = evaluate_classification(y_true, y_proba, t)
        score = m.get(metric, m["f1"])
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def print_classification_report(y_true, y_pred, title: str = "Classification Report"):
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, target_names=["Fully Paid", "Default"]))


def plot_roc_pr_curves(
    y_true,
    y_proba_dict: dict,
    save: bool = True,
    filename: str = "roc_pr_curves.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for name, proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=name, lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, label=name, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {path}")
    plt.close()


def plot_calibration_curve(
    y_true,
    y_proba_dict: Dict[str, np.ndarray],
    n_bins: int = 10,
    save: bool = True,
    filename: str = "calibration_curve.png",
):
    """Reliability diagram for one or more probability sources."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)

    for name, proba in y_proba_dict.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, proba, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=name, lw=2)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (reliability) diagram")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration plot → {path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", save: bool = True):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Fully Paid", "Default"], ax=ax, cmap="Blues"
    )
    ax.set_title(title)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close()
