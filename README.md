# Loan Default Prediction & Credit Risk Intelligence

[![CI](https://github.com/folasunkanmi/loan-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/folasunkanmi/loan-default-prediction/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.1-green.svg)](CHANGELOG.md)

End-to-end credit-risk modelling system for **probability of default (PD)** scoring: dual data schemas, calibrated classifiers, business metrics, SHAP explanations, FastAPI scoring service, Streamlit demo, and drift monitoring.

---

## Business problem

Lenders need an estimate of whether a borrower will default. False negatives (approving a defaulter) drive credit loss; false positives (rejecting a good borrower) drive opportunity cost. This project trains models on either a **Lending Club–style sample** or **synthetic** data, tunes a decision threshold on validation only, and reports **expected financial loss** under a simple cost matrix.

## Dataset

| Source | How to obtain |
|--------|----------------|
| Synthetic (offline demo) | `python -m src.data.generate_data` |
| Real Lending Club sample | `python scripts/download_data.py` |

Schema is detected automatically (`real` vs `synthetic`). The Streamlit form and API **must** match the schema stored in `models/best_pipeline.joblib`.

## Methodology

1. Stratified **train / validation / test** split (test held out completely).
2. Models: Logistic Regression (with SMOTE), Random Forest, XGBoost, LightGBM.
3. Selection by **validation PR-AUC**; threshold by validation F1 (or business loss).
4. **Probability calibration** via `CalibratedClassifierCV` + **`FrozenEstimator`** (sklearn ≥ 1.6) so the base model is not refit during calibration.
5. Final refit on train+val with an internal calibration holdout; **one** evaluation on test.
6. Artifact: pipeline, threshold, schema, feature columns, reference score distribution for drift.

### Example test metrics (synthetic run, seed 42)

These are **illustrative** from a local synthetic train — re-run training for your environment and read `reports/model_comparison.json`:

| Metric | Approx. value |
|--------|----------------|
| ROC-AUC | ~0.90 |
| PR-AUC | ~0.70 |
| Precision / Recall / F1 | depend on threshold |
| Expected loss / loan | cost-matrix dependent |

Do not treat table values as production guarantees.

## Quick start

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.generate_data          # or: python scripts/download_data.py
python -m src.models.train
streamlit run app/streamlit_app.py
# API:
uvicorn api.main:app --reload --port 8000
```

Requires **Python 3.10+** and **scikit-learn ≥ 1.6** (`FrozenEstimator`).

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model metadata |
| GET | `/model/info` | Schema, threshold, test metrics |
| GET | `/metrics` | Live prediction monitor summary |
| POST | `/predict` | Single PD score (+ optional SHAP) |
| POST | `/predict/batch` | Batch scoring |
| POST | `/drift/scores` | PSI-style score drift vs reference |

Interactive docs: `http://localhost:8000/docs`

## SHAP explainability

`src/explainability/shap_analysis.py` unwraps calibrated / frozen estimators, runs TreeExplainer or LinearExplainer on the base classifier, and returns top drivers with **increases_risk / decreases_risk**. Streamlit and `/predict?include_shap=true` surface the same drivers.

## Monitoring

- Prediction JSONL log under `reports/monitoring/`
- `detect_prediction_drift`, `score_prediction_drift_psi`, `detect_feature_drift`
- Streamlit expander: live distribution + drift vs training reference scores

## Testing & CI

```bash
pytest tests/ -v
ruff check src tests api app
```

GitHub Actions (`.github/workflows/ci.yml`): install → ruff → synthetic generate → train smoke → pytest on Python **3.10** and **3.11**.

## Docker

```bash
docker compose up --build
# API :8000  Streamlit :8501
```

## Project layout

```
app/streamlit_app.py     # Interactive scoring UI (schema-locked form)
api/main.py              # FastAPI service
src/models/train.py      # Train + FrozenEstimator calibration
src/evaluation/          # Metrics, calibration plots, business loss
src/explainability/      # SHAP
src/monitoring/          # Logs + PSI drift
src/schemas/             # Pydantic dual-schema validation
tests/
.github/workflows/ci.yml
```

## Deployment notes

Streamlit Community Cloud: point the app at `app/streamlit_app.py`. Cold start may train a small synthetic model via `app/bootstrap.py` if no artifact is present. See [DEPLOYMENT.md](DEPLOYMENT.md).

**Status:** Feature branch is CI-oriented and locally verified for training + tests. Public Cloud URL depends on your Streamlit account deploy — this repo does not claim a live hosted demo unless you deploy one.

## License

MIT — see [LICENSE](LICENSE).

**Version 1.2.1 · November 2026**
