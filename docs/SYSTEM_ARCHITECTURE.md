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
│  ImbPipeline                                               │
│  [ColumnTransformer] → [SMOTE (LR only)] → [Classifier]    │
└────────────────────────────┬───────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   Validation metrics   Threshold search   Model selection
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                    Re-fit on Train+Val
                             │
                             ▼
                    Final Test evaluation
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   best_pipeline.joblib   Metrics & figures   SHAP reports
          │
          ▼
   Streamlit scoring app  (+ optional monitoring)
```

## Components

- **Data layer:** `src/data/`
- **Feature layer:** `src/features/preprocessing.py`
- **Training layer:** `src/models/train.py`
- **Explainability:** `src/explainability/`
- **Serving:** `app/streamlit_app.py`
- **Tests:** `tests/`
