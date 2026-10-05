"""
train_model.py - Machine Learning Model Training Pipeline

Preprocesses loan application data, trains a machine learning classifier,
evaluates performance metrics, and exports model.pkl and scaler.pkl.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(filepath: str = "loans.csv"):
    """Loads and validates loan dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found. Run dataset.py or supply loans.csv first.")
    
    df = pd.read_csv(filepath)
    required_cols = ["income", "credit_score", "loan_amount", "employment_years", "loan_status"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")
    
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Class Distribution:")
    print(df["loan_status"].value_counts(normalize=True).rename({1: "Approved (1)", 0: "Rejected (0)"}))
    return df

def train_and_export(filepath: str = "loans.csv", model_path: str = "model.pkl", scaler_path: str = "scaler.pkl"):
    """Pipeline to train ML classifier, evaluate, and save artifacts."""
    df = load_data(filepath)

    feature_cols = ["income", "credit_score", "loan_amount", "employment_years"]
    X = df[feature_cols]
    y = df["loan_status"]

    # 80/20 Train-Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Preprocessing: StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Machine Learning Classifier: Random Forest
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Train Accuracy : {train_acc * 100:.2f}%")
    print(f"Test Accuracy  : {test_acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved trained model to: '{model_path}'")
    print(f"Saved feature scaler to: '{scaler_path}'")

    # Quick sanity check prediction
    sample_df = pd.DataFrame([[85000, 720, 20000, 5]], columns=feature_cols)
    sample_scaled = scaler.transform(sample_df)
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1]
    print("\nSanity Check Sample: Income=$85k, Credit=720, Loan=$20k, Exp=5yrs")
    print(f"Prediction: {'Approved' if pred == 1 else 'Rejected'} (Confidence: {prob * 100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    train_and_export()
