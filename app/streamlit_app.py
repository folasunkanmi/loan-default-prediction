"""
Interactive Loan Default Prediction App
Run with: streamlit run app/streamlit_app.py

Uses the trained pipeline and, when possible, SHAP for local explanations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import load_best_model

st.set_page_config(
    page_title="Loan Default Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    try:
        return load_best_model()
    except FileNotFoundError:
        return None


artifact = get_model()

st.sidebar.header("Applicant Information")

with st.sidebar.form("applicant_form"):
    st.subheader("Loan Terms")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=12000, step=500)
    term = st.selectbox("Term (months)", options=[36, 60], index=0)

    st.subheader("Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=1000)
    emp_length = st.slider("Employment Length (years)", min_value=0, max_value=11, value=5)
    home_ownership = st.selectbox("Home Ownership", options=["RENT", "MORTGAGE", "OWN", "OTHER"], index=1)
    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "car", "medical",
            "moving", "house", "other",
        ],
        index=0,
    )

    st.subheader("Credit & Risk Indicators")
    grade = st.slider("Grade (numeric proxy, higher = better)", min_value=1.0, max_value=7.0, value=4.5, step=0.1)
    dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=40.0, value=16.0, step=0.5)
    payment_inc_ratio = st.slider("Payment / Income Ratio", min_value=0.5, max_value=30.0, value=8.0, step=0.5)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=12000, step=500)
    revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
    open_acc = st.number_input("Open Credit Lines", min_value=1, max_value=50, value=10)
    delinq_2yrs_zero = st.selectbox(
        "No delinquencies in last 2 years?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    pub_rec_zero = st.selectbox(
        "No public records?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    borrower_score = st.slider("Borrower Score (0–1)", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

    submitted = st.form_submit_button("Predict Default Risk", use_container_width=True)

st.title("💳 Loan Default Prediction & Credit Risk Intelligence")
st.markdown(
    "Enter applicant details in the sidebar and click **Predict Default Risk**. "
    "Powered by a model trained on a real Lending Club sample with SHAP explanations when available."
)

if artifact is None:
    st.error(
        "No trained model found. Please run:\n\n"
        "```bash\npython -m src.models.train\n```"
    )
    st.stop()

pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
model_name = artifact["model_name"]
schema = artifact.get("schema", "real")

st.caption(f"Active model: **{model_name}**  |  Schema: **{schema}**  |  Decision threshold: **{threshold:.3f}**")

if submitted:
    # Build row matching the REAL schema (primary production path)
    row = {
        "loan_amnt": loan_amnt,
        "term": term,
        "annual_inc": annual_inc,
        "dti": dti,
        "payment_inc_ratio": payment_inc_ratio,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "purpose": purpose,
        "home_ownership": home_ownership,
        "delinq_2yrs_zero": delinq_2yrs_zero,
        "pub_rec_zero": pub_rec_zero,
        "open_acc": open_acc,
        "grade": grade,
        "emp_length": emp_length,
        "borrower_score": borrower_score,
    }
    X_new = pd.DataFrame([row])

    try:
        proba = float(pipeline.predict_proba(X_new)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed (schema mismatch?). Details: {e}")
        st.stop()

    prediction = int(proba >= threshold)

    if proba < 0.30:
        tier, color = "Low Risk", "green"
    elif proba < 0.55:
        tier, color = "Medium Risk", "orange"
    else:
        tier, color = "High Risk", "red"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Default Probability", f"{proba:.1%}")
    with col2:
        st.metric("Predicted Class", "Default" if prediction else "Fully Paid")
    with col3:
        st.markdown("**Risk Tier**")
        st.markdown(f":{color}[**{tier}**]")

    st.progress(min(proba, 1.0))

    # --- SHAP local explanation ---
    st.subheader("Top risk drivers (SHAP)")
    try:
        from src.explainability.shap_analysis import compute_shap_values, top_shap_drivers

        explainer, shap_vals, X_trans, feat_names = compute_shap_values(
            pipeline, X_new, max_samples=1
        )
        drivers = top_shap_drivers(shap_vals, feat_names, instance_idx=0, top_k=6)

        for name, val in drivers:
            direction = "increases" if val > 0 else "decreases"
            st.write(f"• **{name}**: SHAP = {val:+.4f} ({direction} default risk)")
    except Exception as e:
        st.info(
            "SHAP explanation unavailable for this prediction "
            f"(fallback to heuristic). Reason: {type(e).__name__}"
        )
        # Heuristic fallback
        drivers = []
        if borrower_score < 0.4:
            drivers.append(f"Low borrower score ({borrower_score:.2f})")
        if dti > 22:
            drivers.append(f"High debt-to-income ({dti:.1f})")
        if revol_util > 70:
            drivers.append(f"High revolving utilization ({revol_util:.0f}%)")
        if payment_inc_ratio > 12:
            drivers.append(f"High payment-to-income ratio ({payment_inc_ratio:.1f})")
        if delinq_2yrs_zero == 0:
            drivers.append("Recent delinquencies present")
        if pub_rec_zero == 0:
            drivers.append("Public records present")
        if grade < 3.5:
            drivers.append(f"Lower grade ({grade:.1f})")
        for d in drivers or ["No major red flags from heuristic rules."]:
            st.write(f"• {d}")

else:
    st.info("← Fill in the applicant form and click **Predict Default Risk** to get a score.")

st.markdown("---")
st.caption(
    "Portfolio project · Real Lending Club data · Logistic Regression · Random Forest · "
    "XGBoost · LightGBM · SHAP · Streamlit · v1.0.0"
)
