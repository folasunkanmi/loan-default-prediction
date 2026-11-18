# Loan Default Prediction & Credit Risk Intelligence

[![CI](https://github.com/folasunkanmi/loan-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/folasunkanmi/loan-default-prediction/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)](CHANGELOG.md)

**Production-oriented ML system for probability of default (PD) scoring, calibration, business metrics, and live monitoring.**

| Field | Detail |
|-------|--------|
| **Domain** | Banking / FinTech / Credit Risk |
| **Task** | Binary classification (default vs paid) |
| **Data** | Real Lending Club sample **or** synthetic fallback |
| **Models** | Logistic Regression · Random Forest · XGBoost · LightGBM |
| **Calibration** | Isotonic / sigmoid (`CalibratedClassifierCV`) |
| **Business metrics** | Precision, Recall, PR-AUC, expected financial loss |
| **Monitoring** | Prediction logs, PSI / KS drift |
| **Interfaces** | Streamlit app + FastAPI REST |
| **CI** | GitHub Actions (pytest + train smoke) |

---

## Quick Start

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data (pick one)
python scripts/download_data.py          # real Lending Club sample
# OR
python -m src.data.generate_data         # synthetic offline

python -m src.models.train
streamlit run app/streamlit_app.py
# API: uvicorn api.main:app --reload --port 8000
```

**Schema rule:** the Streamlit form and API always match the schema stored in `models/best_pipeline.joblib`. Retrain after switching data sources.

---

## Key Features

- Dual schema support (real LC vs synthetic) with Pydantic validation
- Leakage-free Train / Val / Test protocol
- Probability **calibration** + reliability diagrams
- Business metrics: recall, precision, PR-AUC, expected $ loss
- SHAP local explanations
- Model monitoring: score distributions, feature/score PSI drift
- FastAPI: `/predict`, `/predict/batch`, `/health`, `/metrics`
- GitHub Actions CI
- Streamlit Cloud ready (auto-trains synthetic model if artifact missing)

---

## Model performance (illustrative)

Metrics depend on data source and seed. After training, see `reports/model_comparison.json` and calibration plots under `reports/`.

---

## Deploy Streamlit for recruiters

1. Merge/push this branch to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repo: `folasunkanmi/loan-default-prediction`
4. Main file: `app/streamlit_app.py`
5. Deploy → share the public URL.

Cold start may train a small synthetic model via `app/bootstrap.py` if no artifact is present.

More detail: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Project structure

```
app/streamlit_app.py      # Interactive scoring UI
api/main.py               # FastAPI service
src/models/train.py       # Train + calibrate
src/evaluation/metrics.py # Business + calibration metrics
src/monitoring/           # Logs + drift (PSI/KS)
src/schemas/              # Pydantic dual-schema validation
.github/workflows/ci.yml  # CI
tests/
```

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).

**Version 1.2.0 · November 2026**
