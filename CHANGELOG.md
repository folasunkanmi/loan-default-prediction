# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-11-23

### Fixed
- **SHAP explanations for single-applicant predictions**:
  - Root cause: `get_feature_names()` relied on module-level `NUMERIC_FEATURES` /
    `CATEGORICAL_FEATURES` which default to the REAL schema. When the saved model
    was trained on synthetic data, feature names were wrong (length mismatch and
    incorrect labels such as `borrower_score`). The Streamlit app then either
    showed incorrect drivers or fell into the silent exception path.
  - `get_feature_names()` now prefers the fitted `ColumnTransformer.get_feature_names_out()`,
    so names always match the preprocessor that actually produced the features.
  - Streamlit passes the `feature_names` stored in the model artefact when available.
  - Single-row Logistic Regression uses a clearly labelled coefficient × standardised
    feature value approximation (not misrepresented as exact SHAP).
  - Silent `except Exception: return None` replaced with logging of full traceback
    and a user-friendly message; prediction / probability / risk tier always remain visible.
- Risk bands imported from `src.config.settings` (values unchanged: 0.30 / 0.55).

### Added
- Comprehensive SHAP regression tests (`tests/test_shap.py`): feature-name alignment,
  LR single-row approximation, TreeExplainer, dimensionality, failure fallback,
  saved-artefact integration, risk direction.

## [1.1.0] - 2026-11-22

### Improved
- **Streamlit application** completely rewritten for production use:
  - Fully schema-aware (real Lending Club vs synthetic feature sets)
  - Clean professional UI with risk tiers (Low / Medium / High)
  - Displays model name, schema, decision threshold and test ROC-AUC
  - Top SHAP drivers shown for each individual prediction
  - All debug messages and temporary code removed
- Central risk thresholds kept in `src/config/settings.py`
- Version bumped to 1.1.0

### Fixed
- App no longer hard-codes synthetic inputs when a real-data model is loaded
- Prediction results now respect the threshold that was optimised on the validation set

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

[1.1.1]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.1.1
[1.1.0]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.1.0
[1.0.1]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.0.1
[1.0.0]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.0.0
