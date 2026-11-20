# Model Card — Loan Default Prediction (Volume 1)

## Model details

| Field | Value |
|-------|-------|
| Developed by | Folasunkanmi |
| Model date | November 2026 (re-verified 2026-11-20) |
| Version | 1.2.x (integration branch) |
| Type | Calibrated logistic regression inside imbalanced-learn pipeline |
| License | MIT |

## Intended use

- Research / educational PD modelling demonstration  
- Portfolio / interview technical showcase  
- Baseline for further credit-risk work  

**Out of scope:** Production underwriting without institution-specific data, governance, and human review.

## Training data

- Lending Club cleaned sample (`loan_data_lc.csv`), n = 45 342, ~50% default  
- Features treated as available near origination in the cleaned schema  

## Evaluation data

Held-out stratified 20% test set (n = 9 069). Never used for selection or threshold tuning.

## Metrics (test set, verified re-run)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.731 |
| PR-AUC | 0.709 |
| F1 | 0.705 |
| Recall | 0.916 |
| Precision | 0.573 |
| Brier | 0.210 |
| Accuracy | 0.616 |
| Decision threshold | 0.34 |
| Confusion (TN/FP/FN/TP) | 1437 / 3098 / 380 / 4154 |

Source: `reports/real_data_metrics.json` after `python -m src.models.train` with seed 42.

## Ethical considerations

- Public sample; limited fairness analysis (no protected attributes in schema).  
- SHAP attributions are associative.  
- High-stakes decisions require human oversight.

## Caveats

- Balanced sample ≠ production class prior.  
- Calibration and threshold favour high recall (more false positives).  
- Hyper-parameters fixed; further tuning possible.
