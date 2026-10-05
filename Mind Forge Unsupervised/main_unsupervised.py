from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Persona Segmenter API",
    description="FastAPI backend for customer segmentation using K-Means clustering",
    version="1.0.0"
)

# Load trained model artifacts
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
personas = joblib.load("personas.pkl")


class CustomerData(BaseModel):
    annual_income_k: float = Field(
        ...,
        gt=0,
        description="Annual income in thousands"
    )

    spending_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Customer spending score"
    )


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Customer Persona Segmenter API is running"
    }


@app.post("/predict")
def predict_persona(customer: CustomerData):

    input_data = pd.DataFrame([{
        "annual_income_k": customer.annual_income_k,
        "spending_score": customer.spending_score
    }])

    # Scale customer input
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = int(model.predict(input_scaled)[0])

    # Get persona
    persona = personas[cluster]

    return {
        "cluster": cluster,
        "persona": persona,
        "annual_income_k": customer.annual_income_k,
        "spending_score": customer.spending_score
    }