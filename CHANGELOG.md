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

### Changed (production-readiness maintenance)
- Version strings synchronized to **1.1.2** across `src/config/settings.py`, `CITATION.cff`, `pyproject.toml`, README and CHANGELOG.
- `reports/model_comparison.json` includes dataset provenance (synthetic vs real), sample size, default rate, threshold, and environment notes so synthetic metrics are not misread as Lending Club performance.
- Training pipeline writes provenance into both the joblib artefact and the comparison report on every run.

### Documentation
- Added model governance / limitations notes (calibration, fairness, appropriate use).
- Corrected overstated FastAPI claim in earlier changelog entries.

## [1.1.1] - 2026-11-23

### Fixed
- **SHAP explanations for single-applicant predictions** (feature-name alignment, LR approximation labelling, Streamlit path fix).
- Streamlit Cloud `ModuleNotFoundError: No module named 'src'` fixed by adding the repository root to `sys.path`.

### Added
- SHAP and Streamlit import-path regression tests.

## [1.1.0] - 2026-11-22

### Added
- Production Streamlit scoring app for interactive probability-of-default scoring.
  (A separate FastAPI service is not included in this repository release.)
- SHAP explainability module and risk-tier thresholds.
- GitHub Actions CI (pytest on push/PR).

### Changed
- Model training pipeline and artefact format for Lending Club + synthetic schemas.
