# Experiment Log — Volume 1 Real-Data Run (Verified)

**Date of this re-run:** 2026-11-20  
**Branch:** `feature/volume-1-integration`  
**Seed:** 42  
**Calibration:** isotonic via `CalibratedClassifierCV` + `FrozenEstimator` (sklearn ≥ 1.6)

## Data

- File: `data/raw/loan_data_lc.csv` (from `scripts/download_data.py`)
- Observations: **45 342**
- Default rate: **0.500** (balanced sample)
- Schema: **real**

## Split (stratified, seed=42)

| Split | n |
|-------|------|
| Train | 27 204 |
| Validation | 9 069 |
| Test | 9 069 |

Test set was **not** used for model selection or threshold tuning.

## Validation metrics (calibrated probabilities)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
|-------|---------|--------|-----------|--------|-----|-----------|
| logistic_regression | 0.7230 | **0.7060** | 0.580 | 0.884 | 0.7007 | 0.34 |
| random_forest | 0.7277 | 0.7042 | 0.580 | 0.896 | 0.7045 | 0.34 |
| xgboost | 0.7280 | 0.7002 | 0.596 | 0.870 | 0.7072 | 0.35 |
| lightgbm | 0.7233 | 0.6974 | 0.588 | 0.882 | 0.7054 | 0.35 |

**Selected by validation PR-AUC:** `logistic_regression`

## Test metrics (logistic_regression, thr≈0.34)

| Metric | Value |
|--------|-------|
| Accuracy | 0.616 |
| Precision | 0.573 |
| Recall | 0.916 |
| F1 | 0.705 |
| ROC-AUC | 0.731 |
| PR-AUC | 0.709 |
| Brier | 0.210 |
| Expected loss / loan (default costs) | 1322.7 |
| TN / FP / FN / TP | 1437 / 3098 / 380 / 4154 |

## SHAP (mean |SHAP| on 300 test samples)

1. borrower_score  
2. payment_inc_ratio  
3. grade  
4. loan_amnt  
5. term_60  

## Class imbalance handling

- Logistic Regression: `class_weight="balanced"` + **SMOTE** inside the pipeline (train fold only).  
- Tree models: class weights / balanced subsample; no SMOTE.  
- Preprocessor fitted only on training data within each pipeline fit.

## Artefacts

- `models/best_pipeline.joblib`
- `reports/real_data_metrics.json` (source of truth)
- `reports/model_comparison.json`
- `reports/figures/roc_curve.png`
- `reports/figures/pr_curve.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/calibration_curve_test.png`
- `reports/figures/shap_summary.png`
- `docs/VOLUME_1_RESEARCH_PAPER.pdf` (local Pandoc build; binary may need separate push)

## Note on prior Volume 1 write-up

An earlier log on `feature/volume-1-real-data-research` reported **random_forest** as selected (val PR-AUC ≈ 0.711, test ROC-AUC ≈ 0.733) **without** the current FrozenEstimator calibration path. This re-run uses the **integration** training code (calibrated). Numbers above replace that narrative for this branch.
