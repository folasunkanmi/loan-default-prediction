# Technical Documentation

## Overview

This document describes the technical architecture, pipelines, configuration and operational aspects of the Loan Default Prediction system.

## Package Layout

```
src/
├── config/settings.py      # Random seed, paths, constants
├── data/
│   ├── load_data.py        # Real + synthetic loaders
│   └── generate_data.py    # Synthetic fallback generator
├── features/
│   └── preprocessing.py    # Schema detection, ColumnTransformer, prepare_xy
├── models/
│   └── train.py            # Full Train/Val/Test experiment
├── evaluation/
│   └── metrics.py          # Shared metric helpers
└── explainability/
    └── shap_analysis.py    # SHAP utilities
```

## Data Flow

1. `scripts/download_data.py` → `data/raw/loan_data_lc.csv`
2. `load_raw_data()` creates the binary `default` target and selects the modelling columns.
3. `prepare_xy()` + `detect_schema()` choose real vs synthetic feature lists.
4. Stratified split → `data/processed/{train,val,test}.parquet`
5. `ImbPipeline(preprocessor [, SMOTE], classifier)` fitted on train.
6. Threshold optimised on validation.
7. Best model re-fitted on train+val, evaluated once on test.
8. Artefact saved to `models/best_pipeline.joblib`.

## Configuration

- Global seed: `RANDOM_SEED = 42`
- Paths resolved relative to project root via `Path(__file__).resolve().parents[...]`

## Model Artefact Schema

```python
{
  "pipeline": ImbPipeline,
  "model_name": str,
  "threshold": float,
  "schema": "real" | "synthetic",
  "feature_names": list[str],
  "val_results": dict,
  "test_metrics": dict,
}
```

## Streamlit Application

`app/streamlit_app.py` loads the artefact, accepts user inputs matching the feature schema, returns PD, risk tier and optional SHAP force plot.

## Testing

```bash
pytest tests/ -v
```

Covers pipeline construction, schema detection, metric calculation and basic prediction shape.

## Dependencies

See `requirements.txt` and `pyproject.toml`. Core stack: pandas, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap, streamlit, joblib, pyarrow.
