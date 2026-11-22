"""
Loan Default Prediction & Credit Risk Intelligence
Interactive Streamlit scoring application.

Supports both real Lending Club and synthetic model artefacts.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_pipeline.joblib"

# Risk bands (probability of default)
RISK_LOW = 0.30
RISK_MEDIUM = 0.55


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        st.error(
            "Model artefact not found. "
            "Run `python -m src.models.train` from the project root first."
        )
        st.stop()
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        st.error("Unexpected model artefact format. Retrain the model.")
        st.stop()
    return artifact


def get_risk_tier(proba: float) -> Tuple[str, str]:
    """Return (label, streamlit colour) for a default probability."""
    if proba < RISK_LOW:
        return "LOW RISK", "normal"
    if proba < RISK_MEDIUM:
        return "MEDIUM RISK", "off"
    return "HIGH RISK", "inverse"


def try_shap_drivers(
    pipeline, X_row: pd.DataFrame, top_k: int = 5
) -> Optional[List[Tuple[str, float]]]:
    """Attempt to compute top SHAP drivers for a single applicant."""
    try:
        from src.explainability.shap_analysis import compute_shap_values, top_shap_drivers

        explainer, shap_vals, X_trans, feat_names = compute_shap_values(
            pipeline, X_row, max_samples=1
        )
        return top_shap_drivers(shap_vals, feat_names, instance_idx=0, top_k=top_k)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Input widgets – schema aware
# ---------------------------------------------------------------------------
def render_synthetic_inputs() -> Dict[str, Any]:
    st.sidebar.subheader("Applicant Details (Synthetic schema)")
    inputs: Dict[str, Any] = {}
    inputs["loan_amnt"] = st.sidebar.slider("Loan Amount ($)", 1_000, 40_000, 10_000, 500)
    inputs["term"] = st.sidebar.selectbox("Term", [36, 60], format_func=lambda x: f"{x} months")
    inputs["int_rate"] = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 13.0, 0.1)
    inputs["installment"] = st.sidebar.slider("Monthly Installment ($)", 50.0, 1_200.0, 300.0, 10.0)
    inputs["grade"] = st.sidebar.selectbox("Grade", list("ABCDEFG"))
    inputs["emp_length"] = st.sidebar.selectbox(
        "Employment Length",
        [
            "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
            "6 years", "7 years", "8 years", "9 years", "10+ years",
        ],
    )
    inputs["home_ownership"] = st.sidebar.selectbox(
        "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    inputs["annual_inc"] = st.sidebar.number_input(
        "Annual Income ($)", min_value=15_000, max_value=350_000, value=60_000, step=1_000
    )
    inputs["purpose"] = st.sidebar.selectbox(
        "Purpose",
        [
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "other",
        ],
    )
    inputs["dti"] = st.sidebar.slider("Debt-to-Income Ratio", 1.0, 45.0, 15.0, 0.5)
    inputs["delinq_2yrs"] = st.sidebar.slider("Delinquencies (2 yrs)", 0, 8, 0)
    inputs["fico_score"] = st.sidebar.slider("FICO Score", 300, 850, 700, 5)
    inputs["inq_last_6mths"] = st.sidebar.slider("Inquiries (last 6 months)", 0, 10, 1)
    inputs["open_acc"] = st.sidebar.slider("Open Accounts", 1, 30, 10)
    inputs["pub_rec"] = st.sidebar.slider("Public Records", 0, 5, 0)
    inputs["revol_util"] = st.sidebar.slider("Revolving Utilisation (%)", 0.0, 100.0, 40.0, 1.0)
    inputs["total_acc"] = st.sidebar.slider("Total Accounts", 1, 60, 20)
    return inputs


def render_real_inputs() -> Dict[str, Any]:
    st.sidebar.subheader("Applicant Details (Lending Club schema)")
    inputs: Dict[str, Any] = {}
    inputs["loan_amnt"] = st.sidebar.slider("Loan Amount ($)", 500, 40_000, 10_000, 500)
    inputs["term"] = st.sidebar.selectbox("Term", ["36 months", "60 months"])
    inputs["annual_inc"] = st.sidebar.number_input(
        "Annual Income ($)", min_value=10_000, max_value=300_000, value=60_000, step=1_000
    )
    inputs["dti"] = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, 0.5)
    inputs["payment_inc_ratio"] = st.sidebar.slider(
        "Payment / Income Ratio", 0.0, 0.30, 0.08, 0.005
    )
    inputs["revol_bal"] = st.sidebar.number_input(
        "Revolving Balance ($)", min_value=0, max_value=150_000, value=8_000, step=500
    )
    inputs["revol_util"] = st.sidebar.slider("Revolving Utilisation (%)", 0.0, 100.0, 40.0, 1.0)
    inputs["open_acc"] = st.sidebar.slider("Open Accounts", 1, 40, 10)
    inputs["grade"] = st.sidebar.slider("Grade (numeric)", 1.0, 7.0, 3.0, 0.1)
    inputs["emp_length"] = st.sidebar.slider("Employment Length (years)", 0.0, 10.0, 5.0, 0.5)
    inputs["borrower_score"] = st.sidebar.slider("Borrower Score", 0.0, 1.0, 0.5, 0.01)
    inputs["purpose"] = st.sidebar.selectbox(
        "Purpose",
        [
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "other", "car", "medical",
        ],
    )
    inputs["home_ownership"] = st.sidebar.selectbox(
        "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    inputs["delinq_2yrs_zero"] = st.sidebar.selectbox(
        "No delinquencies in last 2 years?", ["Yes", "No"], index=0
    )
    inputs["pub_rec_zero"] = st.sidebar.selectbox(
        "No public records?", ["Yes", "No"], index=0
    )
    # Convert Yes/No to the binary flags the model expects
    inputs["delinq_2yrs_zero"] = 1 if inputs["delinq_2yrs_zero"] == "Yes" else 0
    inputs["pub_rec_zero"] = 1 if inputs["pub_rec_zero"] == "Yes" else 0
    return inputs


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Loan Default Prediction & Credit Risk Intelligence")
    st.caption(
        "Production-oriented probability-of-default model · "
        "Lending Club sample + synthetic fallback · SHAP explainability"
    )

    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "unknown")
    threshold = float(artifact.get("threshold", 0.5))
    schema = artifact.get("schema", "synthetic")
    test_metrics = artifact.get("test_metrics", {})

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Model", model_name.replace("_", " ").title())
    col2.metric("Schema", schema.title())
    col3.metric("Decision Threshold", f"{threshold:.2f}")
    if test_metrics:
        col4.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")

    st.divider()

    # Sidebar inputs
    if schema == "real":
        inputs = render_real_inputs()
    else:
        inputs = render_synthetic_inputs()

    predict_btn = st.sidebar.button("Predict Default Risk", type="primary", use_container_width=True)

    if predict_btn:
        X = pd.DataFrame([inputs])

        try:
            proba = float(pipeline.predict_proba(X)[0, 1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Input row:")
            st.dataframe(X.T, use_container_width=True)
            return

        tier, delta_color = get_risk_tier(proba)
        decision = "Likely to Default" if proba >= threshold else "Likely to Pay"

        # Results
        st.subheader("Risk Assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Default Probability", f"{proba:.1%}")
        m2.metric("Risk Tier", tier)
        m3.metric("Model Decision", decision)

        if proba >= RISK_MEDIUM:
            st.error(f"**{tier}** — Probability of default is elevated. Manual review recommended.")
        elif proba >= RISK_LOW:
            st.warning(f"**{tier}** — Moderate risk. Additional verification may be useful.")
        else:
            st.success(f"**{tier}** — Applicant appears low risk under current model.")

        # SHAP drivers
        drivers = try_shap_drivers(pipeline, X, top_k=6)
        if drivers:
            st.subheader("Top Risk Drivers (SHAP)")
            st.caption("Features that most increased or decreased the predicted default probability for this applicant.")
            driver_df = pd.DataFrame(drivers, columns=["Feature", "SHAP value"])
            driver_df["Direction"] = driver_df["SHAP value"].apply(
                lambda v: "↑ Increases risk" if v > 0 else "↓ Decreases risk"
            )
            st.dataframe(
                driver_df.style.format({"SHAP value": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("SHAP explanation unavailable for this prediction (model or environment limitation).")

        with st.expander("Applicant input summary"):
            st.json(inputs)

    else:
        st.info("Configure the applicant details in the sidebar and click **Predict Default Risk**.")

    # Footer
    st.divider()
    st.caption(
        "Model trained with stratified Train / Validation / Test protocol. "
        "Threshold optimised on validation set only. "
        "See the repository README for methodology, metrics, and citation."
    )


if __name__ == "__main__":
    main()
