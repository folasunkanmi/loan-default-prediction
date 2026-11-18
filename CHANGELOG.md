# Changelog

## [1.2.0] - 2026-11-18

### Added
- Probability calibration (isotonic) and reliability diagrams
- Business metrics: precision, recall, PR-AUC, expected financial loss
- Feature and score drift detection (PSI, KS)
- SHAP explanations for calibrated models (unwrap CalibratedClassifierCV)
- FastAPI `/model/info`, `/drift/scores`, optional SHAP on `/predict`
- Streamlit schema-locked forms + auto-bootstrap for Cloud
- GitHub Actions CI (pytest + train smoke)
- Docker Compose (API + Streamlit)

### Fixed
- Real vs synthetic form/model schema mismatch in Streamlit
- Version string alignment (app, API, pyproject) → 1.2.0
- Threshold always read from trained artifact

## [1.1.0] - 2026-11

- Dual-schema support, FastAPI, basic monitoring, unit tests

## [1.0.0] - 2026-11

- Initial training pipeline, Streamlit app, Lending Club + synthetic data
