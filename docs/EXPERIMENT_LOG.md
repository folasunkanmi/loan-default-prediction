# Experiment Log — Volume 1 Real-Data Runs

**Seed:** 42  
**Branch:** feature/volume-1-integration  

---

## Run A — Volume 1 baseline (pre-calibration artefact on volume-1 branch)

**Data:** `loan_data_lc.csv`, n=45 342, default rate 0.500  
**Split:** Train 27 204 / Val 9 069 / Test 9 069  
**Selected:** random_forest (validation PR-AUC) · threshold 0.38  

| Metric | Test |
|--------|------|
| ROC-AUC | 0.733 |
| PR-AUC | 0.717 |
| F1 | 0.709 |
| Recall | 0.870 |
| Precision | 0.599 |
| Brier | 0.210 |

---

## Run B — Verified retrain 2026-11-20 (integration branch, calibration ON)

**Command:** `python -m src.models.train` with real schema, isotonic calibration via FrozenEstimator  
**Selected by validation PR-AUC:** logistic_regression · threshold ≈ 0.34  

| Model | Val ROC-AUC | Val PR-AUC | Val F1 | thr |
|-------|-------------|------------|--------|-----|
| logistic_regression | 0.723 | **0.706** | 0.701 | 0.34 |
| random_forest | 0.728 | 0.704 | 0.704 | 0.34 |
| xgboost | 0.728 | 0.700 | 0.707 | 0.35 |
| lightgbm | 0.723 | 0.697 | 0.705 | 0.35 |

**Test (logistic_regression, thr≈0.34):**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.731 |
| PR-AUC | 0.709 |
| F1 | 0.705 |
| Recall | 0.916 |
| Precision | 0.573 |
| Brier | 0.210 |
| Accuracy | 0.616 |
| TN/FP/FN/TP | 1437 / 3098 / 380 / 4154 |

**Note:** With calibration enabled, validation PR-AUC ranks Logistic Regression slightly ahead of Random Forest. Discrimination remains in the same band (~0.73 ROC-AUC). Figures under `reports/figures/` correspond to Run B when committed.

Source of truth for Run B: `reports/real_data_metrics.json` and `reports/model_comparison.json`.
