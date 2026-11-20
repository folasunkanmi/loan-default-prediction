# Loan Default Prediction & Credit Risk Intelligence

[![CI](https://github.com/folasunkanmi/loan-default-prediction/actions/workflows/ci.yml/badge.svg?branch=feature/volume-1-integration)](https://github.com/folasunkanmi/loan-default-prediction/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.2-green.svg)](CHANGELOG.md)

**End-to-end credit-risk system:** real Lending Club data · Train/Val/Test modelling · SHAP · Streamlit · FastAPI · monitoring · research documentation.

| Field | Detail |
|-------|--------|
| **Roles demonstrated** | Data Scientist · ML Engineer · Applied AI |
| **Domain** | Banking / FinTech / Credit risk |
| **Task** | Binary PD (probability of default) |
| **Data** | Real Lending Club cleaned sample (~45k) + synthetic offline fallback |
| **Models** | Logistic Regression · Random Forest · XGBoost · LightGBM |
| **Serving** | Streamlit app + FastAPI |
| **Branch** | `feature/volume-1-integration` |

---

## Business problem

Lenders need an estimate of whether a borrower will default. **False negatives** (funding a defaulter) drive credit loss; **false positives** (declining a good borrower) drive opportunity cost. This project trains models under a leakage-free protocol, tunes the decision threshold on validation only, and surfaces scores via UI and API with SHAP explanations and basic drift monitoring.

---

## Volume 1 real-data results (documented)

Logged in [`reports/real_data_metrics.json`](reports/real_data_metrics.json) from the Volume 1 experiment on the public Lending Club sample (seed 42). **Reproduce locally** with `scripts/download_data.py` then `python -m src.models.train`.

**Selected model:** Random Forest · **threshold:** 0.38

| Metric | Test value |
|--------|------------|
| ROC-AUC | 0.733 |
| PR-AUC | 0.717 |
| F1 | 0.709 |
| Recall | 0.870 |
| Precision | 0.599 |
| Brier | 0.210 |

Validation comparison and SHAP drivers are described in [`docs/VOLUME_1_RESEARCH_PAPER.md`](docs/VOLUME_1_RESEARCH_PAPER.md).

> Synthetic-only runs produce different metrics. Do not present synthetic numbers as real-data results.

---

## Quick start

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction
git checkout feature/volume-1-integration

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,api]"

# Real data (preferred for Volume 1)
python scripts/download_data.py
python -m src.models.train

# Or offline synthetic
# python -m src.data.generate_data && python -m src.models.train

streamlit run app/streamlit_app.py
uvicorn api.main:app --reload --port 8000
pytest tests/ -v
```

Requires **Python 3.10+** and **scikit-learn ≥ 1.6** (`FrozenEstimator` for calibration).

---

## Methodology (short)

1. Stratified **Train / Validation / Test** (test never used for selection).
2. Models: LR (+ SMOTE), RF, XGBoost, LightGBM.
3. Select by **validation PR-AUC**; threshold by validation F1.
4. Optional **probability calibration** with `CalibratedClassifierCV` + `FrozenEstimator`.
5. Re-fit on Train+Val; **one** test evaluation.
6. Artefact stores pipeline, threshold, schema, feature columns, reference scores for drift.

---

## Streamlit application

- Form fields driven by trained **schema** (`real` vs `synthetic`).
- **Threshold** always loaded from the artefact (not hardcoded 0.5).
- Input validation via `src/schemas/prediction.py`.
- SHAP top drivers when available.
- Live monitoring expander.

```bash
streamlit run app/streamlit_app.py
```

---

## FastAPI

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model metadata |
| GET | `/model/info` | Schema, threshold, test metrics |
| GET | `/metrics` | Monitor summary |
| POST | `/predict` | Single PD (+ optional SHAP) |
| POST | `/predict/batch` | Batch scoring |
| POST | `/drift/scores` | Score drift vs reference |

Docs: `http://localhost:8000/docs`

---

## Documentation map

| Doc | Purpose |
|------|---------|
| [docs/VOLUME_1_RESEARCH_PAPER.md](docs/VOLUME_1_RESEARCH_PAPER.md) | Research write-up |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Model card |
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Engineering details |
| [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) | Architecture diagram |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Feature definitions |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Reproduce Volume 1 |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Experiment log |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deploy notes |

---

## Project layout

```
app/                 # Streamlit + bootstrap
api/                 # FastAPI
src/
  data/ features/ models/ evaluation/
  explainability/ monitoring/ schemas/ config/
docs/                # Volume 1 research + technical docs
reports/             # metrics JSON (figures after local train)
scripts/download_data.py
tests/
.github/workflows/ci.yml
Dockerfile  docker-compose.yml
```

---

## Testing & CI

```bash
pytest tests/ -v
ruff check src tests api app
```

GitHub Actions: install → synthetic smoke data → ruff → pytest on Python 3.10–3.12.

---

## Limitations (honest)

- Public sample is **balanced**; production books are not.
- Feature set is limited vs modern bureau + alternative data.
- Plot PNGs / PDF paper under `reports/figures/` are generated when you run training locally; this branch ships **metrics JSON and markdown research**, not fabricated image binaries.
- Not a certified credit decisioning system.

---

## License

MIT — see [LICENSE](LICENSE).

**Version 1.2.2 · `feature/volume-1-integration` · November 2026**
