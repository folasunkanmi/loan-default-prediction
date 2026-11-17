# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-11-17

### Fixed
- **Schema alignment**: preprocessing now auto-detects real vs synthetic data and uses matching feature lists (categorical `grade` / `emp_length` on synthetic path).
- **Test-set leakage**: model selection and threshold tuning are performed on a validation set only; the held-out test set is used once for final metrics.
- **SHAP**: LinearExplainer support for Logistic Regression; Streamlit app surfaces top SHAP drivers (with heuristic fallback).
- **Tests**: expanded coverage — preprocessor fit/transform, end-to-end synthetic training, single-applicant prediction, no target leakage.

### Changed
- Training protocol is now Train / Validation / Test (stratified).
- Best model is re-fit on Train+Val before final test evaluation.

## [1.0.0] - 2026-11-17

### Added
- End-to-end loan default prediction pipeline using real Lending Club sample data.
- Model comparison: Logistic Regression, Random Forest, XGBoost, LightGBM.
- Proper `ColumnTransformer` + `Pipeline` preprocessing (no leakage).
- SMOTE + class-weight handling for class imbalance.
- Threshold optimisation maximising F1 on the positive class.
- SHAP explainability utilities (TreeExplainer).
- Interactive Streamlit application for credit risk scoring.
- Synthetic data generator as reproducible fallback.
- Central configuration (`src/config`) for seeds and paths.
- `CITATION.cff` for academic citation.
- Basic pytest smoke tests.
- Production packaging via `pyproject.toml`.

### Dataset
- Primary: cleaned Lending Club sample (~45k loans) derived from public releases
  (via Practical Statistics for Data Scientists repository).
- Fallback: synthetic generator mimicking realistic credit features and default relationships.

[1.0.1]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.0.1
[1.0.0]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.0.0
