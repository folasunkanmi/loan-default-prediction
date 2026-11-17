# Loan Default Prediction & Credit Risk Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.1-green.svg)](CHANGELOG.md)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

**Production-oriented machine learning system for predicting the probability of loan default and supporting credit risk decisions.**

| Field | Detail |
|-------|--------|
| **Domain** | Banking / FinTech / Credit Risk |
| **Task** | Binary classification (Probability of Default) |
| **Primary data** | Real Lending Club sample (~45 k loans) |
| **Models** | Logistic Regression · Random Forest · XGBoost · LightGBM |
| **Explainability** | SHAP (TreeExplainer + LinearExplainer) |
| **Interface** | Streamlit interactive scoring app |
| **Version** | 1.0.1 |

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
- Streamlit app for real-time risk scoring and tiering
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

## Model Performance (real Lending Club sample)

Stratified Train / Validation / Test split. Model selection and threshold tuning on **validation** only; final numbers on held-out **test**.

| Model                  | ROC-AUC | PR-AUC | F1    |
|------------------------|---------|--------|-------|
| **Logistic Regression**| **0.731** | **0.718** | 0.707 |
| Random Forest          | 0.733   | 0.716  | 0.709 |
| XGBoost                | 0.731   | 0.713  | 0.710 |
| LightGBM               | 0.726   | 0.708  | 0.704 |

Exact numbers may vary slightly with environment; all random seeds are fixed via `src/config/settings.py`.

---

## Project Structure

```
loan-default-prediction/
├── app/                     # Streamlit scoring application
├── data/
│   ├── raw/                 # loan_data_lc.csv (via download script)
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── scripts/
│   └── download_data.py     # One-command real dataset download
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── explainability/
├── tests/
├── CITATION.cff
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Reproducibility

- Global random seed: `RANDOM_SEED = 42`
- Stratified Train / Validation / Test split
- Model selection + threshold tuning on validation only
- Test set evaluated once
- SMOTE applied only inside the training pipeline

```bash
pytest tests/ -v
```

---

## Citation

**BibTeX**

```bibtex
@software{folasunkanmi_loan_default_2026,
  author       = {Folasunkanmi},
  title        = {Loan Default Prediction \& Credit Risk Intelligence},
  year         = {2026},
  version      = {1.0.1},
  url          = {https://github.com/folasunkanmi/loan-default-prediction},
  note         = {Machine learning system for probability-of-default modelling on Lending Club data}
}
```

**APA**  
Folasunkanmi. (2026). *Loan Default Prediction & Credit Risk Intelligence* (Version 1.0.1) [Computer software]. https://github.com/folasunkanmi/loan-default-prediction

Please also acknowledge the original Lending Club data source when using the real dataset.

---

## License

MIT License – see [LICENSE](LICENSE).

---

## Acknowledgements

- Lending Club for originally releasing historical loan data
- Nathan George / Peter Gedeck and the Practical Statistics for Data Scientists materials for the cleaned sample
- The open-source communities behind scikit-learn, XGBoost, LightGBM, SHAP, and Streamlit

---

**Version 1.0.1 · November 2026**
