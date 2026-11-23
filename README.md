# Loan Default Prediction & Credit Risk Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.2-green.svg)](CHANGELOG.md)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

**Production-oriented machine learning system for predicting the probability of loan default and supporting credit risk decisions.**

| Field | Detail |
|-------|--------|
| **Domain** | Banking / FinTech / Credit Risk |
| **Task** | Binary classification (Probability of Default) |
| **Primary data** | Real Lending Club sample (~45 k loans) |
| **Models** | Logistic Regression · Random Forest · XGBoost · LightGBM |
| **Explainability** | SHAP (TreeExplainer + LinearExplainer) |
| **Interface** | Streamlit interactive scoring app (schema-aware) |
| **Version** | 1.1.2 |

---

## Abstract

This repository implements a complete, reproducible pipeline for **loan default prediction**. It loads a cleaned sample of historical Lending Club loans, applies leakage-free preprocessing, trains and compares four standard classifiers under class-imbalance handling, optimises the decision threshold on a validation set, and exposes predictions through an interactive Streamlit application. Global and local model explanations are supported via SHAP.

---

## Key Features

- Real public Lending Club data with clear provenance and one-command download
- End-to-end `sklearn` / `imblearn` pipelines (no train–test leakage)
- Proper Train / Validation / Test protocol (model selection on validation only)
- Comparison of linear and tree-based models with consistent metrics
- SMOTE (training only) + class weights
- Threshold tuning for F1 on the default class
- SHAP explanations integrated in the Streamlit app
- Schema-aware Streamlit app with risk tiers and per-applicant drivers
- Synthetic data generator for offline / CI runs
- Centralised configuration, tests, and packaging (`pyproject.toml`)

---

## Installation

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Requires Python 3.11+** (scikit-learn 1.8+ does not support Python 3.10).

---

## Quick Start

```bash
python scripts/download_data.py          # real data (recommended)
# or: python -m src.data.generate_data  # synthetic offline fallback

python -m src.models.train
streamlit run app/streamlit_app.py
```

---

## Dataset

**Primary source:** cleaned Lending Club sample (~45 342 loans).  
**Synthetic fallback:** `generate_loan_dataset` (~50k rows) for CI/offline work.

Always check `reports/model_comparison.json` → `provenance.dataset_type`. Synthetic metrics are **not** real Lending Club performance (typical real ROC-AUC ~0.73).

---

## Model Performance

Stratified Train / Validation / Test. Selection and threshold tuning on **validation** only; final numbers on held-out **test**.

Typical **real** Lending Club sample results (approximate):

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| Logistic Regression | ~0.73 | ~0.72 | ~0.71 |
| Random Forest | ~0.73 | ~0.72 | ~0.71 |
| XGBoost | ~0.73 | ~0.71 | ~0.71 |
| LightGBM | ~0.73 | ~0.71 | ~0.70 |

---

## Risk Tiers vs Decision Threshold

| Concept | Source | Purpose |
|---------|--------|---------|
| Decision threshold | Model artefact (`threshold`, validation F1) | Binary decision in Streamlit |
| Risk bands | `RISK_LOW=0.30`, `RISK_MEDIUM=0.55` | Display tiers only |

---

## Model governance & limitations

See [docs/MODEL_GOVERNANCE.md](docs/MODEL_GOVERNANCE.md) for what has been measured (and what has not): calibration, fairness, regulatory scope, and appropriate use. This release does **not** include FastAPI, monitoring, or automated retraining.

---

## Citation

See [CITATION.cff](CITATION.cff).

---

## License

MIT — see [LICENSE](LICENSE).

---

**Version 1.1.2 · November 2026**
