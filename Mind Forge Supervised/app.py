"""
app.py - Streamlit Frontend Application for Loan Approval Predictor

Provides an interactive user interface for loan applicants/officers to
input financial details, query the prediction engine, and review detailed
approval metrics and risk analysis.
"""

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a polished, modern dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.8rem;
    }
    .result-card-approved {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        color: #065F46;
    }
    .result-card-rejected {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        color: #991B1B;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

def get_credit_tier(score: int):
    if score >= 750:
        return "Excellent", "🟢", "#10B981"
    elif score >= 670:
        return "Good", "🔵", "#3B82F6"
    elif score >= 580:
        return "Fair", "🟠", "#F59E0B"
    else:
        return "Poor", "🔴", "#EF4444"

def check_backend_health():
    """Checks if FastAPI backend is available."""
    try:
        res = requests.get(f"{API_URL}/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None

def run_local_inference(income, credit_score, loan_amount, employment_years):
    """Fallback inference if FastAPI backend is not active."""
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, "Model artifacts (model.pkl / scaler.pkl) not found. Please run train_model.py first."

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features_df = pd.DataFrame([{
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years
        }])
        scaled = scaler.transform(features_df)
        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1])

        lti = round(loan_amount / income, 3)
        confidence = prob if pred == 1 else (1.0 - prob)

        if prob >= 0.75:
            risk = "Low Risk"
            msg = "Applicant meets favorable underwriting criteria with strong approval likelihood."
        elif prob >= 0.50:
            risk = "Moderate Risk"
            msg = "Application conditionally approved; satisfies standard risk thresholds."
        elif prob >= 0.30:
            risk = "High Risk"
            msg = "Application does not satisfy credit or debt-to-income benchmarks."
        else:
            risk = "Critical Risk"
            msg = "Significant risk identified based on credit profile and loan requested."

        return {
            "loan_status": pred,
            "approval_status": "Approved" if pred == 1 else "Rejected",
            "approval_probability": round(prob, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "risk_level": risk,
            "debt_to_income_ratio": lti,
            "message": msg
        }, None
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=64)
    st.title("System Status")
    backend_up, health_data = check_backend_health()

    if backend_up:
        st.success("🟢 FastAPI Backend Online (`port 8000`)")
        mode = "FastAPI Backend"
    else:
        st.info("ℹ️ Backend server offline. Using **Direct Model Inference** mode.")
        mode = "Direct Model"

    st.markdown("---")
    st.markdown("### Architecture Info")
    st.markdown("""
    - **Classifier**: Random Forest Classifier
    - **Preprocessing**: StandardScaler
    - **Target**: `loan_status` (0=Reject, 1=Approve)
    - **Backend**: FastAPI REST API
    - **Frontend**: Streamlit
    """)
    st.markdown("---")
    st.caption("Loan Approval Predictor • v1.0.0")

# Main Page Header
st.markdown('<div class="main-header">🏦 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Supervised Machine Learning Underwriting Engine — evaluate applicant risk & approval in real time</div>', unsafe_allow_html=True)

col_form, col_preview = st.columns([3, 2], gap="large")

with col_form:
    st.subheader("📋 Applicant Information")
    with st.form("loan_application_form"):
        # Form inputs
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Annual Income ($)",
                min_value=10000,
                max_value=500000,
                value=75000,
                step=5000,
                help="Applicant's gross annual verifiable income."
            )
        with c2:
            credit_score = st.slider(
                "Credit Score (FICO)",
                min_value=300,
                max_value=850,
                value=720,
                help="Standard credit score ranging from 300 to 850."
            )
            tier_name, tier_icon, tier_color = get_credit_tier(credit_score)
            st.markdown(f"**Tier:** {tier_icon} <span style='color:{tier_color};font-weight:600;'>{tier_name}</span>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            loan_amount = st.number_input(
                "Requested Loan Amount ($)",
                min_value=2000,
                max_value=200000,
                value=25000,
                step=1000,
                help="Total loan principal requested."
            )
        with c4:
            employment_years = st.slider(
                "Employment History (Years)",
                min_value=0,
                max_value=40,
                value=5,
                help="Continuous years in current profession / employment."
            )

        submit_btn = st.form_submit_btn("🔍 Check Loan Approval", use_container_width=True, type="primary")

with col_preview:
    st.subheader("📊 Financial Health Preview")
    lti = round(loan_amount / income, 3)
    
    # Financial indicators
    st.metric("Loan-to-Income (LTI) Ratio", f"{lti * 100:.1f}%", help="Calculated as Loan Amount / Annual Income")
    
    # Approximate monthly repayment preview (assuming 5 yr loan @ 8.5% p.a.)
    monthly_rate = 0.085 / 12
    n_months = 60
    est_monthly = loan_amount * (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)
    st.metric("Estimated Monthly Payment (60 mo @ 8.5%)", f"${est_monthly:,.2f}")

    if lti > 0.50:
        st.warning("⚠️ High Loan-to-Income ratio: Requested loan exceeds 50% of annual income.")
    elif lti <= 0.30:
        st.success("✅ Favorable Loan-to-Income ratio (≤ 30%).")

# Handle Submission & Prediction
if submit_btn:
    st.markdown("---")
    st.subheader("📝 Underwriting Decision")

    result_data = None
    error_msg = None

    # Step 1: Query API or Local fallback
    if backend_up:
        with st.spinner("Connecting to FastAPI backend for prediction..."):
            try:
                payload = {
                    "income": float(income),
                    "credit_score": int(credit_score),
                    "loan_amount": float(loan_amount),
                    "employment_years": int(employment_years)
                }
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                if res.status_code == 200:
                    result_data = res.json()
                else:
                    error_msg = f"FastAPI Server error ({res.status_code}): {res.text}"
            except Exception as e:
                error_msg = f"Failed to contact API: {e}. Falling back to local model."
                result_data, local_err = run_local_inference(income, credit_score, loan_amount, employment_years)
                if local_err:
                    error_msg += f"\nLocal inference also failed: {local_err}"
    else:
        with st.spinner("Running local ML model inference..."):
            result_data, error_msg = run_local_inference(income, credit_score, loan_amount, employment_years)

    if error_msg and not result_data:
        st.error(f"❌ Could not obtain prediction: {error_msg}")
    elif result_data:
        is_approved = result_data["loan_status"] == 1
        prob = result_data["approval_probability"]
        confidence = result_data["confidence_percentage"]
        risk = result_data["risk_level"]
        message = result_data["message"]

        res_col1, res_col2 = st.columns([3, 2], gap="large")

        with res_col1:
            if is_approved:
                st.markdown(f"""
                <div class="result-card-approved">
                    <h2 style="margin:0;color:#065F46;">✅ APPLICATION APPROVED</h2>
                    <p style="font-size:1.1rem;margin-top:0.5rem;">{message}</p>
                    <p style="margin-bottom:0;"><strong>Approval Confidence:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-rejected">
                    <h2 style="margin:0;color:#991B1B;">❌ APPLICATION REJECTED</h2>
                    <p style="font-size:1.1rem;margin-top:0.5rem;">{message}</p>
                    <p style="margin-bottom:0;"><strong>Rejection Confidence:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Probability Meter")
            st.progress(float(prob))
            st.caption(f"Approval Probability: **{prob * 100:.1f}%** | Rejection Probability: **{(1 - prob) * 100:.1f}%**")

        with res_col2:
            st.markdown("#### Decision Summary")
            st.markdown(f"""
            - **Approval Status**: `{'Approved' if is_approved else 'Rejected'}`
            - **Risk Classification**: `{risk}`
            - **Execution Mode**: `{mode}`
            - **Loan-to-Income (LTI)**: `{lti * 100:.1f}%`
            - **Applicant Credit Tier**: `{tier_name} ({credit_score})`
            """)

            if not is_approved:
                st.info("""
                💡 **Tips to Improve Approval Chance**:
                - Improve credit score above 650+
                - Lower the requested loan amount
                - Increase employment stability (2+ years)
                """)
            else:
                st.success("🎉 Applicant qualifies for competitive institutional interest rates!")
