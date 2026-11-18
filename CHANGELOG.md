# Changelog

## [1.2.1] - 2026-11-18

### Fixed
- `CalibratedClassifierCV(cv="prefit")` replaced with **`FrozenEstimator`** (sklearn ≥ 1.6) so calibration does not refit the base model and is valid on current scikit-learn.
- SHAP unwrapping for calibrated + frozen estimators.
- Monitoring package exports aligned with tests (`detect_prediction_drift`, `score_prediction_drift_psi`, `detect_feature_drift`, `PredictionMonitor`).
- CI: ruff no longer swallowed with `|| true`; scikit-learn floor raised to 1.6.0.

### Tests
- Full suite green locally after calibration fix (see CI for matrix runs).

## [1.2.0] - 2026-11-18

- Calibration plots, business metrics, drift module, dual-schema Streamlit, FastAPI, Docker Compose, GitHub Actions.

## [1.1.0] - 2026-11

- Dual-schema, FastAPI, basic monitoring, unit tests.

## [1.0.0] - 2026-11

- Initial training pipeline and Streamlit app.
