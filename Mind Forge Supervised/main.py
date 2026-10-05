"""
main.py - FastAPI Backend API Server for Loan Approval Predictor

Provides REST API endpoints to predict loan approval using the trained
Random Forest model and StandardScaler artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Loan Approval Predictor API",
    description="Backend API service for real-time loan underwriting risk analysis and approval prediction.",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model file paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Global references for cached model & scaler
model = None
scaler = None

def load_artifacts():
    """Helper to load ML artifacts if present."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
    else:
        print("Warning: model.pkl or scaler.pkl not found. Please run train_model.py first.")

@app.on_event("startup")
def startup_event():
    load_artifacts()

# Pydantic Schemas
class LoanApplication(BaseModel):
    income: float = Field(..., gt=0, description="Annual gross income in USD", example=75000.0)
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)", example=720)
    loan_amount: float = Field(..., gt=0, description="Requested principal loan amount in USD", example=20000.0)
    employment_years: int = Field(..., ge=0, le=60, description="Years in continuous employment", example=5)

class PredictionResponse(BaseModel):
    loan_status: int
    approval_status: str
    approval_probability: float
    confidence_percentage: float
    risk_level: str
    debt_to_income_ratio: float
    message: str

@app.get("/")
def read_root():
    """Root health and discovery endpoint."""
    artifacts_ready = (model is not None) and (scaler is not None)
    return {
        "service": "Loan Approval Predictor API",
        "status": "online",
        "model_loaded": artifacts_ready,
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_ready": (model is not None and scaler is not None)}

@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplication):
    """
    Predict loan approval status based on applicant financial profile.
    """
    global model, scaler
    if model is None or scaler is None:
        # Attempt lazy reload in case training just concluded
        load_artifacts()
        if model is None or scaler is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifacts (model.pkl / scaler.pkl) not loaded. Please train model first."
            )

    try:
        # Prepare input features vector: [income, credit_score, loan_amount, employment_years]
        features_df = pd.DataFrame([{
            "income": application.income,
            "credit_score": application.credit_score,
            "loan_amount": application.loan_amount,
            "employment_years": application.employment_years
        }])

        # Scale features using fitted StandardScaler
        features_scaled = scaler.transform(features_df)

        # Run inference
        prediction = int(model.predict(features_scaled)[0])
        # Probability of class 1 (Approved)
        probabilities = model.predict_proba(features_scaled)[0]
        prob_approved = float(probabilities[1])

        approval_status = "Approved" if prediction == 1 else "Rejected"
        confidence = prob_approved if prediction == 1 else (1.0 - prob_approved)

        # Calculate financial risk assessment
        lti_ratio = round(application.loan_amount / application.income, 3)

        if prob_approved >= 0.75:
            risk_level = "Low Risk"
            message = "Applicant meets favorable underwriting criteria with strong approval likelihood."
        elif prob_approved >= 0.50:
            risk_level = "Moderate Risk"
            message = "Application conditionally approved; satisfies standard risk thresholds."
        elif prob_approved >= 0.30:
            risk_level = "High Risk"
            message = "Application does not satisfy credit or debt-to-income benchmarks."
        else:
            risk_level = "Critical Risk"
            message = "Significant risk identified based on credit profile and loan requested."

        return PredictionResponse(
            loan_status=prediction,
            approval_status=approval_status,
            approval_probability=round(prob_approved, 4),
            confidence_percentage=round(confidence * 100, 2),
            risk_level=risk_level,
            debt_to_income_ratio=lti_ratio,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
