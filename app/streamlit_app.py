"""
Interactive Loan Default Prediction App
Run: streamlit run app/streamlit_app.py

Form fields are driven by artifact['schema'] so real vs synthetic never mismatch.
On Streamlit Cloud, missing model artifacts are trained automatically from synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import ensure_model
from src.models.train import load_best_model
from src.monitoring.tracker import get_monitor
from src.schemas.prediction import validate_and_build_row

st.set_page_config(
    page_title="Loan Default Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    try:
        ensure_model()
        return load_best_model()
    except Exception as exc:
        st.session_state["_model_error"] = str(exc)
        return None


artifact = get_model()

st.title("💳 Loan Default Prediction & Credit Risk Intelligence")

if artifact is None:
    st.error(
        "Could not load or train a model.\n\n"
        f"{st.session_state.get('_model_error', '')}\n\n"
        "Locally run:\n"
        "```bash\npython -m src.data.generate_data\npython -m src.models.train\n```"
    )
    st.stop()

pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])
model_name = artifact["model_name"]
schema = artifact.get("schema", "real")
feature_columns = artifact.get("feature_columns") or []
calibrated = artifact.get("calibrated", False)
test_metrics = artifact.get("test_metrics") or {}

st.markdown(
    "Enter applicant details in the sidebar and click **Predict Default Risk**. "
    f"Active schema is **`{schema}`** — the form only shows compatible fields."
)
st.caption(
    f"Model: **{model_name}** · Schema: **{schema}** · "
    f"Threshold: **{threshold:.3f}** · Calibrated: **{calibrated}**"
)

if test_metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test PR-AUC", f"{test_metrics.get('pr_auc', 0):.3f}")
    c2.metric("Test Recall", f"{test_metrics.get('recall', 0):.3f}")
    c3.metric("Test Precision", f"{test_metrics.get('precision', 0):.3f}")
    c4.metric("Exp. loss / loan", f"${test_metrics.get('expected_loss_per_loan', 0):,.0f}")

st.sidebar.header("Applicant Information")
st.sidebar.success(f"Form locked to schema: `{schema}`")

with st.sidebar.form("applicant_form"):
    if schema == "synthetic":
        st.subheader("Loan Terms")
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 15000, 500)
        term = st.selectbox("Term (months)", [36, 60], index=0)
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
        installment = st.number_input("Installment ($)", 50.0, 2000.0, 350.0, 10.0)

        st.subheader("Borrower Profile")
        annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 72000, 1000)
        grade = st.selectbox("Grade", list("ABCDEFG"), index=2)
        emp_length = st.selectbox(
            "Employment Length",
            [
                "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                "6 years", "7 years", "8 years", "9 years", "10+ years",
            ],
            index=5,
        )
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], 0)
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement",
             "major_purchase", "small_business", "other"],
            0,
        )

        st.subheader("Credit Indicators")
        dti = st.slider("DTI", 0.0, 45.0, 18.0, 0.5)
        fico_score = st.slider("FICO Score", 300, 850, 710, 5)
        revol_util = st.slider("Revolving Util %", 0.0, 120.0, 42.0, 1.0)
        open_acc = st.number_input("Open Accounts", 1, 50, 9)
        total_acc = st.number_input("Total Accounts", 1, 80, 18)
        delinq_2yrs = st.number_input("Delinquencies (2y)", 0, 10, 0)
        pub_rec = st.number_input("Public Records", 0, 10, 0)
        inq_last_6mths = st.number_input("Inquiries (6m)", 0, 15, 1)

        form_data = {
            "loan_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "purpose": purpose,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "fico_score": fico_score,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_util": revol_util,
            "total_acc": total_acc,
        }
    else:
        st.subheader("Loan Terms")
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 12000, 500)
        term = st.selectbox("Term (months)", [36, 60], index=0)

        st.subheader("Borrower Profile")
        annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 65000, 1000)
        emp_length = st.slider("Employment Length (years)", 0, 11, 5)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], 1)
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "small_business", "car", "medical",
                "moving", "house", "other",
            ],
            0,
        )

        st.subheader("Credit Indicators")
        grade = st.slider("Grade (1–7, higher better)", 1.0, 7.0, 4.5, 0.1)
        dti = st.slider("DTI", 0.0, 40.0, 16.0, 0.5)
        payment_inc_ratio = st.slider("Payment / Income", 0.5, 30.0, 8.0, 0.5)
        revol_bal = st.number_input("Revolving Balance ($)", 0, 200000, 12000, 500)
        revol_util = st.slider("Revolving Util %", 0.0, 120.0, 55.0, 1.0)
        open_acc = st.number_input("Open Credit Lines", 1, 50, 10)
        delinq_2yrs_zero = st.selectbox(
            "No delinquencies in last 2 years?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        pub_rec_zero = st.selectbox(
            "No public records?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        borrower_score = st.slider("Borrower Score (0–1)", 0.0, 1.0, 0.55, 0.05)

        form_data = {
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

    submitted = st.form_submit_button("Predict Default Risk", use_container_width=True)

if submitted:
    try:
        X_new = validate_and_build_row(form_data, schema=schema)
    except Exception as e:
        st.error(f"Input validation failed: {e}")
        st.stop()

    if feature_columns:
        missing = [c for c in feature_columns if c not in X_new.columns]
        if missing:
            st.error(f"Schema mismatch. Missing columns: {missing}")
            st.stop()
        X_new = X_new[feature_columns]

    try:
        proba = float(pipeline.predict_proba(X_new)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    prediction = int(proba >= threshold)
    if proba < 0.30:
        tier, color = "Low Risk", "green"
    elif proba < 0.55:
        tier, color = "Medium Risk", "orange"
    else:
        tier, color = "High Risk", "red"

    get_monitor().log(
        probability=proba,
        predicted_class=prediction,
        threshold=threshold,
        model_name=model_name,
        schema=schema,
        risk_tier=tier,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{proba:.1%}")
    col2.metric("Predicted Class", "Default" if prediction else "Fully Paid")
    col3.markdown("**Risk Tier**")
    col3.markdown(f":{color}[**{tier}**]")
    st.progress(min(proba, 1.0))

    st.subheader("Top risk drivers (SHAP)")
    try:
        from src.explainability.shap_analysis import compute_shap_values, top_shap_drivers

        _, shap_vals, _, feat_names = compute_shap_values(pipeline, X_new, max_samples=1)
        drivers = top_shap_drivers(shap_vals, feat_names, instance_idx=0, top_k=6)
        for name, val in drivers:
            direction = "increases" if val > 0 else "decreases"
            st.write(f"• **{name}**: SHAP = {val:+.4f} ({direction} default risk)")
    except Exception as e:
        st.info(f"SHAP unavailable ({type(e).__name__}).")

    with st.expander("Live monitoring"):
        ref = artifact.get("reference_probas")
        st.json(get_monitor().summary(reference_probas=ref))

else:
    st.info("← Fill the form and click **Predict Default Risk**.")

st.markdown("---")
st.caption(
    "Dual-schema · Calibrated probabilities · Business metrics · Drift monitoring · v1.2.0"
)
