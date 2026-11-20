# Reproducibility Guide

## Exact Environment

- Python ≥ 3.10
- Dependencies in `requirements.txt` / `pyproject.toml`

## Steps to Reproduce Volume 1 Results

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
git checkout feature/volume-1-integration

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,app,api]"

# Real data
python scripts/download_data.py

# Full experiment (seed = 42)
python -m src.models.train

pytest tests/ -v
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload --port 8000
```

## Expected Key Numbers (within floating-point tolerance)

- Validation PR-AUC (Random Forest) ≈ 0.711
- Test ROC-AUC ≈ 0.733
- Test F1 ≈ 0.709
- Selected threshold ≈ 0.38

See `reports/real_data_metrics.json` for the logged Volume 1 numbers.

## Random Seeds

All `random_state` / `seed` arguments are set to 42.

## Data Version

The download script fetches a fixed remote file. If the upstream URL changes, the experiment may need updating.
