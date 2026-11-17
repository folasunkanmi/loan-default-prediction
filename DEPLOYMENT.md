# Deployment Guide

## Local API

```bash
# Train (if needed)
python -m src.data.generate_data   # or scripts/download_data.py for real data
python -m src.models.train

# Run FastAPI
pip install fastapi "uvicorn[standard]" pydantic
uvicorn api.main:app --reload --port 8000
```

Open docs: http://localhost:8000/docs

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "real",
    "features": {
      "loan_amnt": 12000,
      "term": 36,
      "annual_inc": 65000,
      "dti": 16.0,
      "payment_inc_ratio": 8.0,
      "revol_bal": 12000,
      "revol_util": 55.0,
      "purpose": "debt_consolidation",
      "home_ownership": "MORTGAGE",
      "delinq_2yrs_zero": 1,
      "pub_rec_zero": 1,
      "open_acc": 10,
      "grade": 4.5,
      "emp_length": 5,
      "borrower_score": 0.55
    }
  }'
```

## Docker

```bash
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

## Live demo options

1. **Render / Railway / Fly.io** – connect the GitHub repo, set start command:
   `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
2. **Hugging Face Spaces** – Docker Space using this `Dockerfile`
3. **Streamlit Community Cloud** – deploy `app/streamlit_app.py` for the interactive UI

Health check path: `/health`  
Monitoring summary: `/metrics`

## Notes

- The API validates inputs with Pydantic against the **trained schema** (`real` or `synthetic`).
- Prediction distributions are logged under `reports/monitoring/predictions.jsonl`.
- For production, mount a persistent volume for `models/` and `reports/` and disable build-time training if you ship a pre-trained artifact.
