# LOAN DEFAULT PREDICTION — VOLUME 1
## REAL-DATA EXPERIMENTS, MODEL EVALUATION AND EXPLAINABLE MACHINE LEARNING

**Author:** Folasunkanmi  
**Repository:** https://github.com/folasunkanmi/loan-default-prediction  
**Branch:** `feature/volume-1-integration`  
**Version:** 1.2.0 (Volume 1 + API/monitoring integration)  
**Date:** November 2026  

---

## Abstract

This volume presents a complete, reproducible real-data experiment for binary loan default prediction using a cleaned sample of historical Lending Club loans (n ≈ 45 342). Four classifiers—Logistic Regression, Random Forest, XGBoost and LightGBM—are trained under a strict Train/Validation/Test protocol with leakage-free preprocessing, class-imbalance handling (SMOTE on training data only for linear models; class weights for tree models), and threshold optimisation performed exclusively on the validation set. The Random Forest model is selected by validation PR-AUC and achieves test-set ROC-AUC 0.733, PR-AUC 0.717 and F1 0.709 (threshold 0.38). SHAP analysis identifies borrower_score, grade, payment_inc_ratio, annual income and loan term as the dominant drivers of predicted default probability. The engineering stack on this branch additionally provides a schema-aware Streamlit application, FastAPI scoring endpoints, and prediction monitoring. All reported real-data metrics are those logged in `reports/real_data_metrics.json`; independent reproduction requires downloading the public sample and re-running training.

**Keywords:** loan default prediction, credit risk, Lending Club, Random Forest, XGBoost, LightGBM, SHAP, explainable AI, class imbalance, threshold optimisation, probability of default, FastAPI, monitoring

---

## 1. Introduction

Accurate estimation of the probability of default (PD) is a core requirement of modern credit risk management. This Volume 1 focuses exclusively on **real data**. Synthetic data is retained only as an offline/CI fallback and is never used to claim experimental results.

---

## 2. Research Problem

Given borrower and loan characteristics known at origination, predict whether a loan will default versus be fully paid, and choose an operating threshold that balances false positives against false negatives.

---

## 3. Research Objectives

1. Data understanding and EDA on the real Lending Club sample.
2. Leakage-free preprocessing and reproducible Train/Validation/Test protocol.
3. Fair comparison of Logistic Regression, Random Forest, XGBoost and LightGBM.
4. Threshold optimisation on validation data only.
5. Comprehensive metrics (including PR-AUC, Brier score).
6. Global and local explanations via SHAP.
7. Business implications, fairness considerations and limitations.
8. Serving and monitoring surfaces consistent with the trained artefact.

---

## 4. Dataset Description and Source

**Source.** Cleaned sample distributed with *Practical Statistics for Data Scientists* materials, derived from public Lending Club loan data.

**Download.** `python scripts/download_data.py` → `data/raw/loan_data_lc.csv`

**Size.** 45 342 observations; default rate 0.50 in this balanced sample (22 671 / 22 671).

**Target.** `default = 1` if outcome is default, else 0.

**Features (real schema).** Numeric: loan_amnt, annual_inc, dti, payment_inc_ratio, revol_bal, revol_util, open_acc, grade, emp_length, borrower_score. Categorical: term, purpose, home_ownership, delinq_2yrs_zero, pub_rec_zero.

---

## 5–9. Methodology (Summary)

- Missing values: median / most-frequent, fitted on train only.
- One-hot encoding; StandardScaler for numerics.
- SMOTE only for Logistic Regression, inside the training pipeline.
- Tree models use class weights.
- Seed 42; stratified ~60/20/20 Train/Val/Test.
- Model selection by validation PR-AUC; threshold by validation F1.
- Final evaluation once on test after re-fit on Train+Val.

---

## 10. Experimental Results

### Validation (model selection)

| Model                  | ROC-AUC | PR-AUC | F1    | Best thr |
|------------------------|---------|--------|-------|----------|
| Logistic Regression    | 0.721   | 0.710  | 0.700 | 0.33     |
| **Random Forest**      | **0.726** | **0.711** | 0.704 | 0.38     |
| XGBoost                | 0.725   | 0.707  | 0.707 | 0.36     |
| LightGBM               | 0.721   | 0.701  | 0.705 | 0.32     |

### Test (Random Forest, thr = 0.38)

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.644  |
| Precision  | 0.599  |
| Recall     | 0.870  |
| F1         | 0.709  |
| ROC-AUC    | 0.733  |
| PR-AUC     | 0.717  |
| Brier      | 0.210  |

Confusion matrix: TN=1892, FP=2643, FN=589, TP=3945.

Source of truth: `reports/real_data_metrics.json`.

**Note on figures.** Plot artefacts under `reports/figures/` are produced when training is run locally with plotting enabled. They are not fabricated placeholders in this integration commit.

---

## 11–12. Model Comparison and SHAP

All four models achieve similar discrimination (ROC-AUC ≈ 0.72–0.73). Random Forest is selected on validation PR-AUC.

Top SHAP drivers (associative, not causal): borrower_score, grade, payment_inc_ratio, annual_inc, term, dti, revol_util.

---

## 13–16. Business Implications, Fairness, Limitations, Recommendations

- Low threshold prioritises recall of defaults.
- Precision ~0.60 implies many flagged loans still perform; secondary rules help.
- No protected attributes in this public sample for full disparate-impact analysis.
- Balanced sample ≠ production imbalance; temporal CV not available in this extract.
- For production: institutional data, temporal validation, calibration, monitoring, human review.

---

## 17. Engineering Surfaces (this integration branch)

- **Streamlit:** schema-aware form, artefact threshold, SHAP drivers, live monitor summary.
- **FastAPI:** `/health`, `/model/info`, `/metrics`, `/predict`, `/predict/batch`, `/drift/scores`.
- **CI:** Python 3.10–3.12, synthetic smoke data, ruff, pytest.

---

## 18. References

1. Lending Club public loan data releases.  
2. Gedeck et al., Practical Statistics for Data Scientists (data companion).  
3. Lundberg & Lee (2017), SHAP.  
4. scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn, FastAPI, Streamlit.  
5. https://github.com/folasunkanmi/loan-default-prediction

---

## 19. Reproducibility

See `docs/REPRODUCIBILITY.md`.

```bash
python scripts/download_data.py
python -m src.models.train
pytest tests/ -v
streamlit run app/streamlit_app.py
uvicorn api.main:app --port 8000
```

---

*End of Volume 1 (integrated)*
