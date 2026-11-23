# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2026-11-23

### Fixed
- GitHub Actions Python 3.10 failures caused by loading a scikit-learn 1.9.1 model artefact under sklearn 1.7.x (latest available on 3.10).

### Changed
- Official minimum Python version is now **3.11** (`requires-python = ">=3.11"`).
  scikit-learn 1.8+ (required to load the current trained pipeline) does not support Python 3.10.
- CI matrix: Python 3.11 and 3.12 only.
- `scikit-learn` lower bound raised to `>=1.8.0` to match the trained artefact environment.
- Artefact SHAP integration test skips cleanly on incompatible sklearn versions.

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
- Streamlit Cloud `ModuleNotFoundError: No module named 'src'` fixed by adding the
  repository root to `sys.path` before any `src.*` imports.

### Added
- Comprehensive SHAP regression tests (`tests/test_shap.py`): feature-name alignment,
  LR single-row approximation, TreeExplainer, dimensionality, failure fallback,
  artefact integration.
- Streamlit import-path regression tests (`tests/test_streamlit_imports.py`).

## [1.1.0] - 2026-11-22

### Added
- Production Streamlit scoring app and FastAPI prediction service.
- SHAP explainability module and risk-tier thresholds.
- GitHub Actions CI (pytest on push/PR).

### Changed
- Model training pipeline and artefact format for Lending Club + synthetic schemas.
