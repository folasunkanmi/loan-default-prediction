# Model artefacts

## Primary file

`best_pipeline.joblib` — trained pipeline (or calibrated wrapper) plus metadata:

- `pipeline`, `model_name`, `threshold`, `schema`
- `feature_columns`, `test_metrics`, `val_results`
- `reference_probas` / `reference_stats` for drift monitoring

## How to create

```bash
# Preferred: real Lending Club sample
python scripts/download_data.py
python -m src.models.train

# Offline: synthetic
python -m src.data.generate_data
python -m src.models.train
```

Streamlit cold-start can also call `app.bootstrap.ensure_model()` which trains a synthetic model if the artefact is missing.

## Note on Git

Large joblib files may be omitted from some branches. Always retrain after clone for the schema you need. Volume 1 real-data metrics are recorded in `reports/real_data_metrics.json`.
