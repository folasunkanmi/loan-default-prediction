# System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data       │────▶│  Preprocessing   │────▶│  Train/Val/Test │
│  (LC sample)    │     │  + Schema detect │     │  stratified     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                     ┌─────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│  ImbPipeline (+ optional calibration)                      │
│  [ColumnTransformer] → [SMOTE (LR only)] → [Classifier]    │
└────────────────────────────┬───────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   Validation metrics   Threshold search   Model selection
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                    Re-fit on Train+Val (+ calibrate)
                             │
                             ▼
                    Final Test evaluation
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   best_pipeline.joblib   Metrics & figures   SHAP reports
          │
          ├──────────────────┬──────────────────┐
          ▼                  ▼                  ▼
   Streamlit app         FastAPI            Monitoring
   (schema-aware)        /predict           (drift + logs)
```

## Components

| Layer | Location |
|-------|----------|
| Data | `src/data/`, `scripts/download_data.py` |
| Features | `src/features/preprocessing.py` |
| Training | `src/models/train.py` |
| Evaluation | `src/evaluation/metrics.py` |
| Explainability | `src/explainability/shap_analysis.py` |
| Schemas | `src/schemas/prediction.py` |
| Monitoring | `src/monitoring/` |
| Streamlit | `app/streamlit_app.py`, `app/bootstrap.py` |
| API | `api/main.py` |
| Tests | `tests/` |
| CI | `.github/workflows/ci.yml` |
| Docs / research | `docs/` |
