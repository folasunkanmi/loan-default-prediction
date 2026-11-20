---
title: "Volume 1 — Loan Default Prediction on Lending Club Data"
author: "Folasunkanmi"
date: "2026-11-20"
---

# Abstract

This report documents a reproducible experiment for binary loan-default prediction on a cleaned Lending Club sample (n = 45,342). Four classifiers were trained under a stratified train/validation/test protocol with isotonic probability calibration via `FrozenEstimator`. Model selection used validation PR-AUC only. The selected model on this re-run is **logistic_regression** with test ROC-AUC **0.731**, PR-AUC **0.709**, and F1 **0.705** at threshold **0.34**.

# 1. Introduction

Lenders estimate probability of default (PD) to balance credit loss against opportunity cost. This volume focuses on methodology integrity: isolation of the test set, explicit imbalance handling, calibration, and transparent metrics.

# 2. Data

- Source: cleaned Lending Club sample via `scripts/download_data.py`
- Rows: 45,342; default rate: 0.500; schema: `real`
- Split (seed 42): train 27,204 / val 9,069 / test 9,069

# 3. Methods

1. Stratified 60/20/20-style split (20% test, 20% of original as val from remainder).
2. Models: logistic regression (balanced + SMOTE), random forest, XGBoost, LightGBM.
3. Fit on train; calibrate on a train-internal or validation holdout using **isotonic** `CalibratedClassifierCV(FrozenEstimator(...))`.
4. Thresholds maximise validation F1; best model maximises **validation PR-AUC**.
5. Final model refit on train+val with calibration holdout; **one** test evaluation.
6. SHAP on the unwrapped base estimator for interpretability.

# 4. Results

## 4.1 Validation (selection)

| Model | ROC-AUC | PR-AUC | F1 | Threshold |
|-------|---------|--------|-----|-----------|
| logistic_regression | 0.7230 | 0.7060 | 0.7007 | 0.34 |
| random_forest | 0.7277 | 0.7042 | 0.7045 | 0.34 |
| xgboost | 0.7280 | 0.7002 | 0.7072 | 0.35 |
| lightgbm | 0.7233 | 0.6974 | 0.7054 | 0.35 |

Selected: **logistic_regression**.

## 4.2 Test (final)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7309 |
| PR-AUC | 0.7090 |
| Precision | 0.5728 |
| Recall | 0.9162 |
| F1 | 0.7049 |
| Accuracy | 0.6165 |
| Brier | 0.2103 |
| Threshold | 0.34 |
| TN / FP / FN / TP | 1437 / 3098 / 380 / 4154 |

## 4.3 Figures

![ROC](reports/figures/roc_curve.png)

![Precision-Recall](reports/figures/pr_curve.png)

![Confusion matrix](reports/figures/confusion_matrix.png)

![Calibration](reports/figures/calibration_curve_test.png)

![SHAP summary](reports/figures/shap_summary.png)

# 5. Limitations

Balanced labels, limited feature set, and a recall-oriented threshold increase false positives. Results are not production underwriting guidance.

# 6. Reproducibility

See `docs/REPRODUCIBILITY.md`. Canonical metrics: `reports/real_data_metrics.json`.

# References

1. Lending Club public loan data (cleaned sample used in *Practical Statistics for Data Scientists*).
2. scikit-learn documentation: calibration, `FrozenEstimator`.
3. Lundberg & Lee, SHAP (2017).
