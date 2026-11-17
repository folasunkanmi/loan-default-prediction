"""Evaluation helpers and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, precision_recall_curve, roc_curve,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def print_classification_report(y_true, y_pred, title: str = "Classification Report"):
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, target_names=["Fully Paid", "Default"]))


def plot_roc_pr_curves(y_true, y_proba_dict: dict, save: bool = True, filename: str = "roc_pr_curves.png"):
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
