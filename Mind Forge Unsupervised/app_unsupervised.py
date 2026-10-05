"""
app_unsupervised.py
Project: Customer Persona Segmenter (Unsupervised)
Purpose: Streamlit dashboard - collects a new customer's income and
         spending score, sends them to the FastAPI backend
         (main_unsupervised.py) to get their cluster/persona, and plots
         them on a 2D scatter plot against the existing clusters.

Run with:
    python -m streamlit run app_unsupervised.py

Expects the FastAPI backend (main_unsupervised.py) to already be running at:
    http://localhost:8000/predict

Expected request/response contract with the backend
(confirm this matches whatever main_unsupervised.py actually implements,
and edit the field names below if the backend teammate names things
differently):

    POST http://localhost:8000/predict
    Request body (JSON):
        {
            "annual_income_k": float,
            "spending_score": float
        }
    Response body (JSON):
        {
            "cluster": int,
            "persona": str
        }

Also expects "customers_with_clusters.csv" (produced by train_kmeans.py)
to be in the same folder, so the existing clusters can be plotted.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000/predict"
CLUSTERED_DATA_PATH = "customers_with_clusters.csv"

st.set_page_config(page_title="Customer Persona Segmenter", page_icon="🧩", layout="centered")

st.title("🧩 Customer Persona Segmenter")
st.write("Enter a customer's income and spending score to see which persona they belong to.")

with st.form("customer_form"):
    annual_income_k = st.slider("Annual Income (in $1000s)", min_value=0, max_value=150, value=60)
    spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    submitted = st.form_submit_button("Find Persona")

new_customer = None

if submitted:
    payload = {
        "annual_income_k": annual_income_k,
        "spending_score": spending_score,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        cluster = result.get("cluster")
        persona = result.get("persona")

        st.divider()
        st.success(f"This customer belongs to: **{persona}** (Cluster {cluster})")

        new_customer = {
            "annual_income_k": annual_income_k,
            "spending_score": spending_score,
            "cluster": cluster,
        }

    except requests.exceptions.ConnectionError:
        st.warning(
            "⚠️ Could not reach the backend server. Make sure main_unsupervised.py is running:\n\n"
            "`python -m uvicorn main_unsupervised:app --reload --port 8000`"
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ---------------------------------------------------------
# Cluster visualization
# ---------------------------------------------------------
st.divider()
st.subheader("Cluster Map")

try:
    df = pd.read_csv(CLUSTERED_DATA_PATH)

    color_key = "persona" if "persona" in df.columns else "cluster"

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in df.groupby(color_key):
        ax.scatter(group["annual_income_k"], group["spending_score"], label=str(label), alpha=0.6, s=40)

    if new_customer is not None:
        ax.scatter(
            new_customer["annual_income_k"],
            new_customer["spending_score"],
            color="black",
            marker="*",
            s=400,
            label="New Customer",
            edgecolors="white",
            linewidths=1.5,
        )

    ax.set_xlabel("Annual Income (k)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Clusters")
    ax.legend()
    st.pyplot(fig)

except FileNotFoundError:
    st.info(
        f"'{CLUSTERED_DATA_PATH}' not found. Run train_kmeans.py first to generate it, "
        "then the cluster map will appear here."
    )

st.divider()
st.caption("Model: K-Means (k=3) | Features: annual income, spending score")
