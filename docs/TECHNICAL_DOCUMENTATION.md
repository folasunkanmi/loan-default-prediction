# Technical Documentation

## Overview

End-to-end loan default prediction system: real Lending Club data, Train/Val/Test modelling, SHAP, Streamlit UI, FastAPI serving, and basic prediction monitoring.

## Package Layout

```
src/
├── config/settings.py
├── data/
├── features/preprocessing.py
├── models/train.py
├── evaluation/metrics.py
├── explainability/shap_analysis.py
├── monitoring/
│   ├── tracker.py
│   └── drift.py
└── schemas/prediction.py

api/main.py
app/streamlit_app.py
app/bootstrap.py
```

## Data Flow

1. `scripts/download_data.py` → `data/raw/loan_data_lc.csv`
2. `load_raw_data()` creates binary `default` and modelling columns
3. `prepare_xy()` + `detect_schema()` select real vs synthetic feature lists
4. Stratified Train/Val/Test → processed parquet splits
5. `ImbPipeline` fitted on train; threshold on validation; test once
6. Artefact → `models/best_pipeline.joblib`

## Model Artefact Schema (API branch)

```python
{
  "pipeline": fitted model (pipeline or calibrated wrapper),
  "model_name": str,
  "threshold": float,
  "schema": "real" | "synthetic",
  "feature_columns": list[str],
  "feature_names": list[str],
  "val_results": dict,
  "test_metrics": dict,
  "calibrated": bool,
  "reference_stats": dict,
  "reference_probas": list[float],
}
```

## Streamlit

- Loads threshold, schema, and feature columns from the artefact
- Validates inputs via `validate_and_build_row`
- Logs predictions to the monitor
- Optional SHAP drivers via `explain_instance`

## FastAPI

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness + model status |
| `/model/info` | GET | Metadata and test metrics |
| `/metrics` | GET | Monitor summary |
| `/predict` | POST | Single score (+ SHAP) |
| `/predict/batch` | POST | Batch scores |
| `/drift/scores` | POST | Prediction drift vs reference |

```bash
uvicorn api.main:app --reload --port 8000
```

## Testing & CI

```bash
pytest tests/ -v
ruff check src tests api app
```

CI (`.github/workflows/ci.yml`): matrix Python 3.10–3.12, synthetic smoke data, ruff, pytest.
