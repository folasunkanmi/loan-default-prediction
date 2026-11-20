# Experiment Log — Volume 1 Real-Data (verified retrain)

**Date:** 2026-11-20  
**Branch:** feature/volume-1-integration  
**Seed:** 42  
**Calibration:** isotonic via CalibratedClassifierCV + FrozenEstimator  

## Data

- File: `data/raw/loan_data_lc.csv`
- Observations: 45 342
- Default rate: 0.500
- Schema: real

## Split

- Train: 27 204
- Validation: 9 069
- Test: 9 069

## Validation metrics (calibrated)

| Model | ROC-AUC | PR-AUC | F1 | Threshold |
|-------|---------|--------|-----|-----------|
| **logistic_regression** | 0.723 | **0.706** | 0.701 | 0.34 |
| random_forest | 0.728 | 0.704 | 0.704 | 0.34 |
| xgboost | 0.728 | 0.700 | 0.707 | 0.35 |
| lightgbm | 0.723 | 0.697 | 0.705 | 0.35 |

**Selected:** logistic_regression (highest validation PR-AUC under calibrated protocol)

## Test metrics (logistic_regression, thr=0.34)

- Accuracy: 0.616
- Precision: 0.573
- Recall: 0.916
- F1: 0.705
- ROC-AUC: 0.731
- PR-AUC: 0.709
- Brier: 0.210
- TN / FP / FN / TP: 1437 / 3098 / 380 / 4154

## SHAP top features (mean |SHAP|)

1. borrower_score
2. payment_inc_ratio
3. grade
4. loan_amnt
5. term_60 / term_36
6. purpose_debt_consolidation
7. dti

## Artefacts

- `models/best_pipeline.joblib` (retrain after clone)
- `reports/real_data_metrics.json`
- `reports/calibration_*.png`, `reports/roc_pr_curves_*.png`, `reports/shap_summary.png` (generated during retrain)

## Note

Earlier Volume 1 documentation reported Random Forest as champion **without** the same calibration path. This verified retrain selects Logistic Regression by validation PR-AUC when calibration is enabled. Discrimination remains similar (~0.73 ROC-AUC).
