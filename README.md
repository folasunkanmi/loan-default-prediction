# Loan Default Prediction & Credit Risk Intelligence

[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.3-green.svg)](CHANGELOG.md)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

**Educational machine-learning demonstration for loan-default prediction using a balanced, curated Lending Club sample.**

| Field | Detail |
|-------|--------|
| **Domain** | Banking / FinTech / Credit Risk |
| **Task** | Binary classification (Probability of Default) |
| **Primary data** | Curated Lending Club sample (45,342 loans; balanced classes) |
| **Models** | Logistic Regression · Random Forest · XGBoost · LightGBM |
| **Explainability** | SHAP (TreeExplainer + LinearExplainer) |
| **Interface** | Streamlit interactive scoring app (schema-aware) |
| **Version** | 1.1.3 |

---

## Abstract

This repository implements a reproducible demonstration pipeline for **loan default classification**. It loads a cleaned, balanced sample of historical Lending Club loans, fits preprocessing on training data, compares four classifiers using validation data, selects a decision threshold on validation data, and exposes predictions through Streamlit. SHAP explanations are available for the saved model.

---

## Key Features

- Curated public Lending Club sample with recorded source and SHA-256 provenance
- End-to-end `sklearn` / `imblearn` pipelines (no train–test leakage)
- Proper Train / Validation / Test protocol (model selection on validation only)
- Comparison of linear and tree-based models using validation PR-AUC
- Training-only SMOTE when the training partition is imbalanced
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

**Supported Python versions: 3.11 and 3.12.** (scikit-learn 1.8+ does not support Python 3.10.)

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

**Primary source:** [cleaned Lending Club sample](https://github.com/gedeck/practical-statistics-for-data-scientists/raw/master/data/loan_data.csv.gz), containing 45,342 loans. It is a **balanced curated sample** (22,671 defaults and 22,671 paid-off loans), not a representative population sample.
**Synthetic fallback:** `generate_loan_dataset` (~50k rows) for CI/offline work.

The target is constructed from `outcome` (`default` = 1, `paid off` = 0). `borrower_score` is excluded because its point-in-time availability is unclear and it could encode target information. Check `reports/model_comparison.json` → `provenance` for the dataset checksum, counts, and training environment.

---

## Model Performance

Stratified Train / Validation / Test. Selection and threshold tuning on **validation** only; final numbers on held-out **test**.

Measured metrics are produced by `python -m src.models.train` and recorded in [`reports/model_comparison.json`](reports/model_comparison.json), along with the untouched test-set confusion matrix and ROC/PR plot. See the [Phase 1 evaluation report](reports/PHASE1_REAL_DATA_EVALUATION_REPORT.md) for the exact run and interpretation. These results describe only this balanced curated sample and do **not** estimate real-world lending performance.

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

**Version 1.1.3 · November 2026**
