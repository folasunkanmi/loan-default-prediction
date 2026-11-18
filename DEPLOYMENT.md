# Deployment Guide

## 1. Local

```bash
pip install -r requirements.txt
python -m src.data.generate_data          # or scripts/download_data.py
python -m src.models.train
streamlit run app/streamlit_app.py
# API:
uvicorn api.main:app --reload --port 8000
```

## 2. Streamlit Community Cloud (recommended for recruiters)

1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** → select `folasunkanmi/loan-default-prediction`.
4. Branch: `main` (or `feature/api-monitoring-tests-schema` until merged).
5. Main file path: `app/streamlit_app.py`
6. Advanced settings → Python 3.11.

**Important for Cloud:** the app needs a trained model artifact.

Option A – commit a small synthetic-trained model (demo):

```bash
python -m src.data.generate_data
python -m src.models.train
# ensure models/best_pipeline.joblib is not gitignored for the demo
# or train in a bootstrap script on startup (slower cold start)
```

Option B – add a bootstrap in Streamlit that trains if missing (see README).

After deploy, share the public URL (e.g. `https://loan-default-prediction.streamlit.app`).

## 3. Docker

```bash
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

## 4. Render / Railway

Start command:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Health check: `/health`

## Schema note

Streamlit and the API **always** use the schema stored in `models/best_pipeline.joblib`.
Retrain on real data (`scripts/download_data.py`) or synthetic (`src.data.generate_data`)
before scoring. Mixing forms and models causes the missing-columns error.
