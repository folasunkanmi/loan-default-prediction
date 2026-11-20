# Model Card — Loan Default Prediction (Random Forest)

## Model Details

- **Developed by:** Folasunkanmi
- **Model date:** November 2026
- **Model version:** 1.2.0 (Volume 1 integration)
- **Model type:** Random Forest Classifier (scikit-learn) inside an imbalanced-learn pipeline, with optional probability calibration
- **License:** MIT
- **Contact:** via GitHub repository

## Intended Use

- Research and educational demonstration of probability-of-default modelling.
- Portfolio / technical interview showcase.
- Baseline for further credit-risk research.
- API and Streamlit scoring demos with monitoring hooks.

**Out of scope:** Direct production underwriting without institution-specific retraining, calibration, governance and human oversight.

## Training Data

- Real Lending Club cleaned sample (n = 45 342, balanced 50 % default).
- Features known at or near origination.

## Evaluation Data

Held-out stratified 20 % test set (n ≈ 9 069), never used for model or threshold selection.

## Metrics (Test Set — Volume 1 reported)

| Metric    | Value |
|-----------|-------|
| ROC-AUC   | 0.733 |
| PR-AUC    | 0.717 |
| F1        | 0.709 |
| Recall    | 0.870 |
| Precision | 0.599 |
| Brier     | 0.210 |
| Threshold | 0.38  |

## Ethical Considerations

- No protected attributes present in the public sample; fairness analysis limited.
- Explanations are associative only.
- High-stakes decisions require human review.

## Caveats

- Balanced sample does not reflect production class imbalance.
- Limited feature set compared with modern bureau + alternative data models.
- Fixed hyper-parameters; further tuning possible.
