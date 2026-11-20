# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-20

### Added
- **Volume 1 Research Paper** (`docs/VOLUME_1_RESEARCH_PAPER.md`) with full real-data experiment write-up.
- Comprehensive technical documentation suite:
  - `docs/TECHNICAL_DOCUMENTATION.md`
  - `docs/SYSTEM_ARCHITECTURE.md`
  - `docs/MODEL_CARD.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/DATA_DICTIONARY.md`
  - `docs/EXPERIMENT_LOG.md`
- Real-data evaluation artefacts: `reports/real_data_metrics.json`, confusion matrix, ROC, PR, calibration and SHAP figures under `reports/figures/`.
- Confirmed end-to-end real Lending Club experiment (Random Forest selected, test ROC-AUC ≈ 0.733, F1 ≈ 0.709).

### Changed
- README updated with links to Volume 1 and documentation, actual measured results, and clearer reproducibility instructions.

## [1.0.1] - 2026-11-17

### Fixed
- **Schema alignment**: preprocessing now auto-detects real vs synthetic data and uses matching feature lists.
- **Test-set leakage**: model selection and threshold tuning on validation only.
- **SHAP** and tests expanded.

### Changed
- Training protocol is now Train / Validation / Test (stratified).

## [1.0.0] - 2026-11-17

### Added
- End-to-end loan default prediction pipeline using real Lending Club sample data.
- Model comparison, pipelines, SMOTE, threshold optimisation, SHAP, Streamlit app, synthetic fallback, tests, packaging.

[1.1.0]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.1.0
[1.0.1]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.0.1
[1.0.0]: https://github.com/folasunkanmi/loan-default-prediction/releases/tag/v1.0.0
