# Loan Default Prediction & Credit Risk Intelligence

**Industry:** Banking / FinTech  
**Level:** Advanced Portfolio Project  
**Target Roles:** Data Scientist · Machine Learning Engineer · AI Engineer

End-to-end machine learning system that predicts the probability of loan default using a **real Lending Club sample** and provides explainable credit risk insights.

## Key Features

- Real public Lending Club loan data (~45k loans)
- Full cleaning & feature engineering pipeline
- Model comparison: **Logistic Regression · Random Forest · XGBoost · LightGBM**
- Class imbalance handling (SMOTE + class weights)
- Evaluation with Precision, Recall, F1, ROC-AUC, PR-AUC + threshold optimisation
- SHAP explainability utilities
- Interactive **Streamlit** prediction application
- Modular, production-ready code structure

## Dataset

This project uses a cleaned sample of historical Lending Club loans (originally from the public Lending Club data releases, curated in the [Practical Statistics for Data Scientists](https://github.com/gedeck/practical-statistics-for-data-scientists) repository).

- ~45,342 loans
- Balanced target (`default` / `paid off`)
- Features: loan amount, term, income, DTI, revolving utilisation, grade, employment length, purpose, home ownership, etc.

**Attribution:** Data derived from Lending Club public loan data. Please respect any original terms of use.

A synthetic data generator is also included as a fallback (`python -m src.data.generate_data`).

## Project Structure

```
loan-default-prediction/
├── data/
│   ├── raw/                 # place loan_data_lc.csv here
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── explainability/
├── app/                     # Streamlit app
├── models/                  # saved pipelines
├── reports/
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/folasunkanmi/loan-default-prediction.git
cd loan-default-prediction

# 2. Create & activate environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure data is present
# Download the real dataset (see data/raw/README.md) or generate synthetic:
# python -m src.data.generate_data

# 5. Train models
python -m src.models.train

# 6. Launch the interactive app
streamlit run app/streamlit_app.py
```

## Model Performance (real Lending Club sample)

| Model                | ROC-AUC | PR-AUC | F1     |
|----------------------|---------|--------|--------|
| **Logistic Regression** | **0.731** | **0.718** | 0.707 |
| Random Forest        | 0.733   | 0.716  | 0.709 |
| XGBoost              | 0.731   | 0.713  | 0.710 |
| LightGBM             | 0.726   | 0.708  | 0.704 |

*(Results from stratified 80/20 split; threshold optimised for F1)*

## Techniques Demonstrated

- Proper stratified train/test split
- `ColumnTransformer` + `Pipeline` (no leakage)
- SMOTE applied only on training data
- Threshold tuning for business-relevant decisions
- SHAP TreeExplainer ready
- Model serialisation with joblib
- Clean, modular package structure suitable for production

## Business Framing

This system supports:
- Automated risk tiering (Low / Medium / High)
- Explainable decisions useful for compliance discussions
- Rapid what-if analysis for underwriters
- Portfolio-level risk monitoring foundation

## License

MIT License – see [LICENSE](LICENSE).

Feel free to use this project in your portfolio, fork it, and extend it.

---

**Built as an advanced portfolio piece for Data Science / ML Engineer roles.**
