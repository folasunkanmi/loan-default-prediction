# Reproducibility Guide

## Exact Environment

- Python ≥ 3.10
- Dependencies pinned in spirit by `requirements.txt` (install latest compatible).

## Steps to Reproduce Volume 1 Results

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
git checkout feature/volume-1-real-data-research   # or main after merge

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Real data
python scripts/download_data.py

# Full experiment (seed = 42)
python -m src.models.train

# Optional: generate extra figures / SHAP
# (scripts or notebooks under notebooks/)

pytest tests/ -v
```

## Expected Key Numbers (within floating-point tolerance)

- Validation PR-AUC (Random Forest) ≈ 0.711
- Test ROC-AUC ≈ 0.733
- Test F1 ≈ 0.709
- Selected threshold ≈ 0.38

## Random Seeds

All `random_state` / `seed` arguments are set to 42.

## Data Version

The download script fetches a fixed remote file. If the upstream URL changes, the experiment may need updating; the current file hash can be recorded by users for strict pinning.
