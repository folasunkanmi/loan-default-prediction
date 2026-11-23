# Phase 1: Real Lending Club Sample Evaluation

## Summary

The Phase 1 pipeline was rerun on the repository's tracked Lending Club sample
with a stratified train/validation/test protocol. XGBoost was selected by
validation PR-AUC, its decision threshold was selected by validation F1, and
the final model was evaluated once on the untouched test partition.

**Important limitation:** this is a balanced curated sample, not a
representative lending population. The reported metrics do not estimate
real-world Lending Club performance, default prevalence, calibration, or
underwriting outcomes.

## Dataset and provenance

| Property | Value |
|---|---|
| Upstream sample | [Practical Statistics for Data Scientists Lending Club sample](https://github.com/gedeck/practical-statistics-for-data-scientists/raw/master/data/loan_data.csv.gz) |
| Tracked file | `data/raw/loan_data_lc.csv` |
| SHA-256 (CSV) | `b41c02cbf45e07ca93bfa0e5af2e322fb084a4576777e70e7dbddc8ceb87ea37` |
| Rows | 45,342 |
| Paid off (target 0) | 22,671 |
| Default (target 1) | 22,671 |
| Default prevalence in this curated sample | 50% |

The target is constructed only from `outcome`: `default` maps to 1 and
`paid off` maps to 0. Unknown or missing labels are rejected. Status and other
source columns are not model inputs. `borrower_score` is excluded because its
point-in-time availability is unclear and it could encode target information.

The numeric features are loan amount, annual income, debt-to-income ratio,
payment-to-income ratio, revolving balance, revolving utilisation, open
accounts, grade, and employment length. Term, purpose, home ownership, and the
two zero-event flags are treated as categorical features. Numeric inputs are
median-imputed and scaled; categorical inputs are most-frequent-imputed and
one-hot encoded. Preprocessing is fitted within each training pipeline and the
same fitted transformations are used for prediction.

## Evaluation protocol

- Seed: `42`; stratified split: 60% train / 20% validation / 20% test.
- Split sizes: 27,204 train; 9,069 validation; 9,069 test.
- Four candidate models were fit on train only. Model selection used validation
  PR-AUC; threshold selection maximised validation F1. The test data did not
  participate in either choice.
- The selected model was refit on train plus validation, then evaluated on the
  held-out test partition once. SMOTE was not applied because the sample is
  balanced.
- Precision, recall, and F1 use the validation-selected threshold. ROC-AUC and
  PR-AUC use probability scores.

### Validation results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Selected threshold |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6904 | 0.6731 | 0.5552 | 0.9200 | 0.6925 | 0.33 |
| Random Forest | 0.6981 | 0.6782 | 0.5565 | 0.9361 | 0.6980 | 0.34 |
| XGBoost | **0.6995** | **0.6808** | 0.5654 | 0.9144 | 0.6987 | 0.33 |
| LightGBM | 0.6958 | 0.6751 | 0.5598 | **0.9336** | **0.7000** | 0.31 |

XGBoost was selected by PR-AUC, not by the maximum F1.

### Untouched test results

| Selected model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.33 | 0.7084 | 0.6855 | 0.5655 | 0.9007 | 0.6948 |

| Test confusion matrix (actual rows, predicted columns) | Predicted paid off | Predicted default |
|---|---:|---:|
| Actual paid off | 1,397 | 3,138 |
| Actual default | 450 | 4,084 |

The test-set plots are [`roc_pr_curves_test.png`](roc_pr_curves_test.png) and
[`confusion_matrix.png`](confusion_matrix.png). Machine-readable metrics and
full provenance are in [`model_comparison.json`](model_comparison.json); the
loadable pipeline, selected threshold, feature names, and metadata are in
`models/best_pipeline.joblib`.

## Reproduction and limits

With the tracked sample present, run:

```bash
python -m src.models.train
```

The measured run recorded in `model_comparison.json` used Python 3.15.2 and
scikit-learn 1.9.1 on Windows. The project's supported runtimes remain Python
3.11 and 3.12; CI is configured to exercise both. This report does not claim
calibration, fairness validation, regulatory compliance, or production
readiness. The sample's balanced selection means its scores and threshold must
not be transferred to a real lending population.
