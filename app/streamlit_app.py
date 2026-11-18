import streamlit as st
import pandas as pd
import joblib
import pathlib

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction & Credit Risk Intelligence")
st.write("Fixed version - works with synthetic data")

model_path = pathlib.Path("models/best_pipeline.joblib")
if not model_path.exists():
    st.error("Model not found!")
    st.stop()

loaded = joblib.load(model_path)
# Handle if saved as dict
if isinstance(loaded, dict):
    # try common keys
    pipeline = loaded.get('model') or loaded.get('pipeline') or loaded.get('best_pipeline') or loaded.get('estimator')
    if pipeline is None:
        # take first value that has predict_proba
        for v in loaded.values():
            if hasattr(v, 'predict_proba'):
                pipeline = v
                break
    st.write(f"Loaded dict with keys: {list(loaded.keys())}")
else:
    pipeline = loaded

if not hasattr(pipeline, 'predict_proba'):
    st.error(f"Still no predict_proba. Type: {type(pipeline)}")
    st.stop()

st.success(f"Active model loaded: {model_path} | Schema: synthetic")

st.sidebar.header("Applicant Details")
inputs = {}
inputs['loan_amnt'] = st.sidebar.slider("Loan Amount", 1000, 40000, 10000)
inputs['term'] = st.sidebar.selectbox("Term", ["36 months", "60 months"])
inputs['int_rate'] = st.sidebar.slider("Interest Rate %", 5.0, 30.0, 13.0)
inputs['installment'] = st.sidebar.slider("Installment", 50.0, 1000.0, 300.0)
inputs['grade'] = st.sidebar.selectbox("Grade", ["A","B","C","D","E","F","G"])
inputs['emp_length'] = st.sidebar.selectbox("Emp Length", ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"])
inputs['home_ownership'] = st.sidebar.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE"])
inputs['annual_inc'] = st.sidebar.number_input("Annual Income", 20000, 200000, 60000)
inputs['purpose'] = st.sidebar.selectbox("Purpose", ["debt_consolidation","credit_card","home_improvement","other"])
inputs['dti'] = st.sidebar.slider("DTI", 0.0, 40.0, 15.0)
inputs['open_acc'] = st.sidebar.slider("Open Acc", 1, 20, 10)
inputs['pub_rec'] = st.sidebar.slider("Public Records", 0, 5, 0)
inputs['revol_util'] = st.sidebar.slider("Revol Util %", 0.0, 100.0, 50.0)
inputs['total_acc'] = st.sidebar.slider("Total Acc", 1, 40, 20)
inputs['inq_last_6mths'] = st.sidebar.slider("Inquiries last 6m", 0, 10, 1)
inputs['delinq_2yrs'] = st.sidebar.slider("Delinq 2yrs", 0, 10, 0)
inputs['fico_score'] = st.sidebar.slider("FICO Score", 600, 850, 700)

if st.sidebar.button("Predict Default Risk", type="primary"):
    df = pd.DataFrame([inputs])
    try:
        proba = pipeline.predict_proba(df)
        score = float(proba[0,1]) if proba.shape[1]==2 else float(proba[0])
        st.metric("Default Probability", f"{score:.2%}")
        if score > 0.5:
            st.error("HIGH RISK - Likely to Default")
        else:
            st.success("LOW RISK - Likely to Pay")
        st.balloons()
        st.json(inputs)
    except Exception as e:
        st.error(f"Error: {e}")
        st.write(df.T)
