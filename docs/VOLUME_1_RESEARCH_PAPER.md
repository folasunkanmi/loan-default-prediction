# LOAN DEFAULT PREDICTION — VOLUME 1
## REAL-DATA EXPERIMENTS, MODEL EVALUATION AND EXPLAINABLE MACHINE LEARNING

**Author:** Folasunkanmi  
**Repository:** https://github.com/folasunkanmi/loan-default-prediction  
**Version:** 1.1.0 (Volume 1)  
**Date:** November 2026  

---

## Abstract

This volume presents a complete, reproducible real-data experiment for binary loan default prediction using a cleaned sample of historical Lending Club loans (n ≈ 45 342). Four classifiers—Logistic Regression, Random Forest, XGBoost and LightGBM—are trained under a strict Train/Validation/Test protocol with leakage-free preprocessing, class-imbalance handling (SMOTE on training data only for linear models; class weights for tree models), and threshold optimisation performed exclusively on the validation set. The Random Forest model is selected by validation PR-AUC and achieves test-set ROC-AUC 0.733, PR-AUC 0.717 and F1 0.709 (threshold 0.38). SHAP analysis identifies borrower_score, grade, payment_inc_ratio, annual income and loan term as the dominant drivers of predicted default probability. The study emphasises proper evaluation methodology, calibration awareness, business cost of false positives versus false negatives, and responsible-AI considerations. All code, metrics and figures are released for full reproducibility.

**Keywords:** loan default prediction, credit risk, Lending Club, Random Forest, XGBoost, LightGBM, SHAP, explainable AI, class imbalance, threshold optimisation, probability of default

---

## 1. Introduction

Accurate estimation of the probability of default (PD) is a core requirement of modern credit risk management. Regulatory frameworks (Basel, IFRS 9) and commercial lending decisions both depend on reliable, well-calibrated and interpretable models. Public datasets such as historical Lending Club loan tapes provide a realistic test bed for applied machine-learning research while remaining accessible for academic and portfolio work.

This Volume 1 focuses exclusively on **real data**. Synthetic data is retained only as an offline/CI fallback and is never used to claim experimental results.

---

## 2. Research Problem

Given a set of borrower and loan characteristics known at origination, predict whether a loan will ultimately default (or be charged off) versus be fully paid. The practical decision is not only ranking risk but choosing an operating threshold that balances the cost of declining good borrowers (false positives) against the cost of funding borrowers who later default (false negatives).

---

## 3. Research Objectives

1. Perform thorough data understanding and exploratory analysis on the real Lending Club sample.
2. Implement leakage-free preprocessing and a reproducible Train/Validation/Test protocol.
3. Train and fairly compare Logistic Regression, Random Forest, XGBoost and LightGBM.
4. Optimise the classification threshold on validation data only.
5. Deliver comprehensive evaluation metrics (including PR-AUC, Brier score and calibration).
6. Provide global and local explanations via SHAP.
7. Discuss business implications, fairness considerations and limitations.
8. Release all artefacts so that an independent researcher can reproduce the numbers.

---

## 4. Dataset Description and Source

**Source.** Cleaned sample distributed with the materials of *Practical Statistics for Data Scientists* (Gedeck et al.), itself derived from public Lending Club loan data releases.

**Download.**  
```bash
python scripts/download_data.py
```
produces `data/raw/loan_data_lc.csv` (≈ 7.3 MB).

**Size after loading & cleaning.** 45 342 observations, 15 modelling features + binary target.

**Target definition.**  
`default = 1` if `outcome == "default"`, else 0.  
In this balanced sample the default rate is exactly 0.50 (22 671 defaults / 22 671 paid-off).

**Key features (real schema).**  
- Numeric: `loan_amnt`, `annual_inc`, `dti`, `payment_inc_ratio`, `revol_bal`, `revol_util`, `open_acc`, `grade`, `emp_length`, `borrower_score`  
- Categorical: `term` (36 / 60), `purpose`, `home_ownership`, `delinq_2yrs_zero`, `pub_rec_zero`

**Attribution.** Users must respect the original terms of use of Lending Club public data and the Practical Statistics distribution.

---

## 5. Exploratory Data Analysis (Summary of Findings)

- The sample is deliberately balanced (50 % default), which simplifies metric interpretation but does **not** reflect the natural class imbalance of production lending books.
- Default rate declines strongly with better (higher) numeric grade and higher borrower_score.
- 60-month term loans exhibit higher default rates than 36-month loans.
- Purpose “small_business” shows elevated default risk relative to “credit_card” or “debt_consolidation”.
- `payment_inc_ratio`, `dti` and `revol_util` are positively associated with default.
- Correlation analysis confirms expected relationships (grade ↔ borrower_score, payment_inc_ratio ↔ dti) without extreme multicollinearity that would preclude tree models.

Figures are stored under `reports/figures/` (`target_distribution.png`, `default_by_grade.png`, `default_by_term.png`, `default_by_purpose.png`, `correlation_heatmap.png`).

---

## 6. Data Preprocessing Methodology

1. **Missing values.** Median imputation for numeric, most-frequent for categorical (fitted on training data only).
2. **Duplicates.** None material after the upstream cleaning.
3. **Categorical encoding.** One-hot encoding with `handle_unknown="ignore"`.
4. **Scaling.** StandardScaler on numeric features (required for Logistic Regression; neutral for trees).
5. **Outliers.** No aggressive winsorisation; tree models are robust and extreme values carry signal.
6. **Feature selection.** All available features in the cleaned schema are retained; dimensionality after one-hot is modest (~32).
7. **Class imbalance.**  
   - Logistic Regression: SMOTE (k_neighbors=5) applied **only inside the training pipeline**.  
   - Tree models: `class_weight="balanced"` / `"balanced_subsample"`.
8. **Leakage prevention.** All transformers are fitted exclusively on the training fold; validation and test receive only `transform`.

---

## 7. Experimental Design

- **Random seed:** 42 (global, documented in `src/config/settings.py` and training code).
- **Split:** Stratified 60 % / 20 % / 20 % Train / Validation / Test.
- **Model selection criterion:** Validation PR-AUC (primary) with secondary attention to F1 and ROC-AUC.
- **Threshold tuning:** Grid search over [0.15, 0.75] maximising F1 on the **validation** set only.
- **Final evaluation:** Single pass on the untouched test set after model and threshold are locked.
- **Re-fit:** Best model is re-trained on Train+Validation before test evaluation.

---

## 8. Model Development

| Model                | Key hyper-parameters (fixed for reproducibility) |
|----------------------|--------------------------------------------------|
| Logistic Regression  | max_iter=2000, class_weight=balanced, solver=lbfgs |
| Random Forest        | n_estimators=250, max_depth=12, min_samples_leaf=15, class_weight=balanced_subsample |
| XGBoost              | n_estimators=400, max_depth=6, lr=0.05, subsample=0.85, colsample_bytree=0.8 |
| LightGBM             | n_estimators=500, max_depth=7, lr=0.05, num_leaves=63, class_weight=balanced |

All models are wrapped in `imblearn.pipeline.Pipeline` together with the preprocessor (and SMOTE for LR).

---

## 9. Evaluation Methodology

Metrics reported for every model on validation (selection) and for the champion on test:

- Accuracy, Precision, Recall, F1 (positive class = default)
- ROC-AUC, PR-AUC
- Brier score
- Confusion matrix (TN, FP, FN, TP)
- Calibration curve
- Threshold sensitivity

Business interpretation:  
- **False Negative** (missed default) → direct credit loss.  
- **False Positive** (declined good loan) → opportunity cost / customer friction.  
In this experiment the operating point is deliberately recall-oriented (high sensitivity to default).

---

## 10. Experimental Results

### Validation performance (model selection)

| Model                  | ROC-AUC | PR-AUC | F1    | Best thr |
|------------------------|---------|--------|-------|----------|
| Logistic Regression    | 0.721   | 0.710  | 0.700 | 0.33     |
| **Random Forest**      | **0.726** | **0.711** | 0.704 | 0.38     |
| XGBoost                | 0.725   | 0.707  | 0.707 | 0.36     |
| LightGBM               | 0.721   | 0.701  | 0.705 | 0.32     |

Random Forest selected by highest validation PR-AUC.

### Final test-set performance (Random Forest, thr = 0.38)

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.644  |
| Precision  | 0.599  |
| Recall     | 0.870  |
| F1         | 0.709  |
| ROC-AUC    | 0.733  |
| PR-AUC     | 0.717  |
| Brier      | 0.210  |

Confusion matrix (test):  
TN = 1 892, FP = 2 643, FN = 589, TP = 3 945.

Figures: `reports/figures/roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `calibration.png`.

---

## 11. Model Comparison

All four models achieve similar discrimination (ROC-AUC ≈ 0.72–0.73). Tree ensembles slightly outperform logistic regression on PR-AUC and F1 after threshold tuning. Random Forest offers the best validation PR-AUC and competitive test metrics while remaining relatively easy to explain and deploy.

---

## 12. Explainability and SHAP Analysis

TreeExplainer applied to the final Random Forest (sample of 400 test instances).

**Top features by mean |SHAP| (default class):**

1. `borrower_score`  
2. `grade`  
3. `payment_inc_ratio`  
4. `annual_inc`  
5. `term_60` / `term_36`  
6. `dti`  
7. `revol_util`

Higher borrower_score and better grade push predicted PD downward; higher payment-to-income ratio, longer term and higher DTI push it upward. These associations are consistent with domain knowledge but **do not establish causation**.

Global summary and bar plots are saved as `reports/figures/shap_summary.png` and `shap_importance.png`.

---

## 13. Business Implications

- Operating at a relatively low threshold (0.38) prioritises recall of defaults, suitable when the cost of a missed default is high relative to the cost of a false decline.
- Precision of ~0.60 implies that roughly 40 % of loans flagged as high-risk will still perform; further policy rules or secondary underwriting may be warranted.
- Calibration is reasonable but not perfect (Brier 0.21); probability estimates should be treated as ranking scores unless further calibrated.

---

## 14. Fairness and Responsible AI

The present dataset does not contain protected attributes (race, gender, age, etc.) that would allow a full disparate-impact analysis. Subgroup performance by purpose, home ownership and term is feasible and should be monitored in any production deployment. Model explanations must be presented as associative, not causal. Human review remains essential for high-stakes decisions.

---

## 15. Limitations

1. The sample is artificially balanced; real portfolios are heavily imbalanced.
2. Features are limited to those present in the cleaned public sample; modern credit models use richer bureau and alternative data.
3. No temporal (issue-date) validation was performed because reliable origination dates are not retained in this particular extract.
4. Hyper-parameters were fixed rather than exhaustively tuned; further gains are possible.
5. SHAP values describe model behaviour, not real-world causal effects.

---

## 16. Recommendations

- For production, retrain on the institution’s own historical data with proper temporal cross-validation and natural class imbalance.
- Apply probability calibration (Platt / isotonic) if well-calibrated PDs are required for capital or pricing.
- Continuously monitor data drift and prediction drift.
- Combine the model score with business rules and human underwriting.
- Extend the feature set and explore fairness metrics once protected attributes are available under appropriate governance.

---

## 17. Conclusion

Volume 1 delivers a transparent, reproducible baseline for loan-default prediction on real Lending Club data. Random Forest emerges as a strong, explainable champion with test ROC-AUC 0.733 and F1 0.709 under a recall-oriented threshold. The accompanying code, metrics, figures and documentation form a solid foundation for further research (Volume 2) and for portfolio demonstration.

---

## 18. References

1. Lending Club. Historical loan data (public releases).  
2. Gedeck, P., et al. *Practical Statistics for Data Scientists* (data companion).  
3. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.  
4. scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn documentation.  
5. Project repository: https://github.com/folasunkanmi/loan-default-prediction

---

## 19. Reproducibility Instructions

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py          # real data
python -m src.models.train               # full experiment
pytest tests/ -v
streamlit run app/streamlit_app.py
```

All random seeds are fixed. Metrics and figures under `reports/` correspond to the run documented in this volume.

---

*End of Volume 1*
