# Experiment Log — Volume 1 Real-Data Run

**Date:** 2026-11-20  
**Branch:** feature/volume-1-real-data-research  
**Seed:** 42  

## Data

- File: `data/raw/loan_data_lc.csv`
- Observations: 45 342
- Default rate: 0.500
- Schema: real

## Split

- Train: 27 204
- Validation: 9 069
- Test: 9 069

## Validation Metrics

| Model               | ROC-AUC | PR-AUC | F1    | Threshold |
|---------------------|---------|--------|-------|-----------|
| logistic_regression | 0.7210  | 0.7099 | 0.7001| 0.33      |
| random_forest       | 0.7256  | 0.7106 | 0.7038| 0.38      |
| xgboost             | 0.7253  | 0.7072 | 0.7065| 0.36      |
| lightgbm            | 0.7211  | 0.7009 | 0.7049| 0.32      |

**Selected:** random_forest (highest PR-AUC)

## Test Metrics (Random Forest, thr=0.38)

- Accuracy: 0.644
- Precision: 0.599
- Recall: 0.870
- F1: 0.709
- ROC-AUC: 0.733
- PR-AUC: 0.717
- Brier: 0.210
- TN / FP / FN / TP: 1892 / 2643 / 589 / 3945

## SHAP Top Features (mean |SHAP|)

1. borrower_score
2. grade
3. payment_inc_ratio
4. annual_inc
5. term_60 / term_36
6. dti
7. revol_util

## Artefacts

- `models/best_pipeline.joblib`
- `reports/model_comparison.json`
- `reports/real_data_metrics.json`
- `reports/figures/*.png`
