# Reproducibility Guide (Volume 1)

## Environment

- Python ≥ 3.10
- `scikit-learn >= 1.6` (`FrozenEstimator`)
- Install: `pip install -r requirements.txt`

## Steps

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
git checkout feature/volume-1-integration

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python -m src.models.train

pytest tests/ -v
```

## Expected key numbers (this integration pipeline, seed 42)

Floating-point tolerance applies; **source of truth is** `reports/real_data_metrics.json` from your run.

- Selected model: **logistic_regression** (by validation PR-AUC)
- Validation PR-AUC (LR) ≈ **0.706**
- Test ROC-AUC ≈ **0.731**
- Test PR-AUC ≈ **0.709**
- Test F1 ≈ **0.705**
- Threshold ≈ **0.34**

Prior docs on `feature/volume-1-real-data-research` reported random_forest without the current calibration stack; do not mix the two narratives.

## Seeds

All `random_state` / seeds default to **42**.
