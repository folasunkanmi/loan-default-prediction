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

This repository implements a complete, reproducible pipeline for **loan default prediction**. It loads a cleaned sample of historical Lending Club loans, applies leakage-free preprocessing, trains and compares four standard classifiers under class-imbalance handling, optimises the decision threshold on a validation set, and exposes predictions through an interactive application. Global and local model explanations are supported via SHAP. The codebase is structured for both research reproducibility and portfolio / production demonstration.

---

## Key Features

- Real public Lending Club data with clear provenance and one-command download
- End-to-end `sklearn` / `imblearn` pipelines (no train–test leakage)
- Proper Train / Validation / Test protocol (model selection on validation only)
- Comparison of linear and tree-based models with consistent metrics
- SMOTE (training only) + class weights
- Threshold tuning for F1 on the default class
- SHAP explanations (TreeExplainer + LinearExplainer) integrated in the app
- **Schema-aware Streamlit app** with risk tiers and per-applicant SHAP drivers
- Synthetic data generator for fully offline / CI runs
- Centralised configuration, tests, and packaging (`pyproject.toml`)
- Machine-readable citation (`CITATION.cff`)

---

## Installation

```bash
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or: pip install -e ".[app,dev]"
```

**Requires Python 3.11+** (scikit-learn 1.8+, used by the trained model artefact, does not support Python 3.10).

---

## Quick Start

```bash
# 1. Get the real dataset (recommended)
python scripts/download_data.py

#    Offline fallback (no internet needed for data):
#    python -m src.data.generate_data

# 2. Train all models (Train/Val/Test protocol, no test leakage)
python -m src.models.train

# 3. Launch the interactive scoring application
streamlit run app/streamlit_app.py
```

The Streamlit app automatically detects whether the saved model was trained on the real Lending Club schema or the synthetic schema and presents the matching input fields.

---

## Dataset

**Primary source**  
Cleaned sample of Lending Club loans (~45 342 observations) derived from public Lending Club data releases and distributed via the [Practical Statistics for Data Scientists](https://github.com/gedeck/practical-statistics-for-data-scientists) repository.

| Attribute | Value |
|-----------|-------|
| Observations | ~45 342 |
| Target | `default` (1) / paid off (0) — balanced in the sample |
| Key features | loan amount, term, annual income, DTI, revolving utilisation, grade, employment length, purpose, home ownership, borrower score, etc. |

**One-command download**

```bash
python scripts/download_data.py
```

This saves `data/raw/loan_data_lc.csv`.

**Attribution**  
Data ultimately originates from Lending Club public loan data. Users should respect the original terms of use of any upstream source.

**Synthetic fallback**  
`python -m src.data.generate_data` creates a statistically realistic synthetic dataset (default 50 k rows) for offline development, testing, and continuous integration. The training code auto-detects which schema is present.

---

## Model Performance

Stratified Train / Validation / Test split. Model selection and threshold tuning on **validation** only; final numbers on held-out **test**.

### Typical results on real Lending Club sample

| Model                  | ROC-AUC | PR-AUC | F1    |
|------------------------|---------|--------|-------|
| Logistic Regression    | ~0.73   | ~0.72  | ~0.71 |
| Random Forest          | ~0.73   | ~0.72  | ~0.71 |
| XGBoost                | ~0.73   | ~0.71  | ~0.71 |
| LightGBM               | ~0.73   | ~0.71  | ~0.70 |

Exact numbers vary slightly with environment and random seeds (fixed via `src/config/settings.py`). After training, see `reports/model_comparison.json` for the precise metrics of the current run.

---

## Project Structure

```text
loan-default-prediction/
├── app/streamlit_app.py          # Interactive scoring UI
├── src/
│   ├── config/settings.py        # Version, risk bands, seeds
│   ├── data/                     # Load / generate data
│   ├── features/preprocessing.py # Leakage-free transforms
│   ├── models/train.py           # Train / select / save pipeline
│   └── explainability/           # SHAP drivers
├── models/best_pipeline.joblib   # Saved champion pipeline
├── tests/                        # pytest suite
├── requirements.txt
└── pyproject.toml
```

---

## Risk Tiers

Default probability bands (centralised in `src.config.settings`):

| Tier | Probability range |
|------|-------------------|
| LOW RISK | < 0.30 |
| MEDIUM RISK | 0.30 – 0.55 |
| HIGH RISK | ≥ 0.55 |

---

## Citation

See [CITATION.cff](CITATION.cff). Prefer citing the repository URL and version tag.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- Lending Club public loan data and community datasets
- The open-source communities behind scikit-learn, XGBoost, LightGBM, SHAP, and Streamlit

---

**Version 1.1.2 · November 2026**
