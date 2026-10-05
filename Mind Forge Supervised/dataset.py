"""
dataset.py - Synthetic Loan Applicant Data Generator

Generates realistic applicant data (income, credit score, loan amount, employment history)
and exports it to loans.csv for machine learning training.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic applicant dataset with realistic financial distributions.
    """
    np.random.seed(random_seed)

    # 1. Income: between $20,000 and $150,000
    income = np.random.randint(20000, 150000, size=num_samples)

    # 2. Credit Score: standard FICO range 300 to 850
    credit_score = np.random.randint(300, 850, size=num_samples)

    # 3. Loan Amount: between $5,000 and $50,000
    loan_amount = np.random.randint(5000, 50000, size=num_samples)

    # 4. Employment Years: 0 to 20 years
    employment_years = np.random.randint(0, 20, size=num_samples)

    # 5. Loan Approval Rule (with realistic domain logic):
    # - Low credit score (<580) usually leads to rejection
    # - High credit score (>700) with reasonable debt-to-income leads to approval
    # - Employment stability and income boost approval probability
    loan_status = []
    for inc, cs, la, ey in zip(income, credit_score, loan_amount, employment_years):
        lti_ratio = la / inc
        score = 0.0

        # Credit score weighting
        if cs >= 720:
            score += 0.50
        elif cs >= 620:
            score += 0.25
        elif cs < 550:
            score -= 0.40

        # Loan to Income weighting
        if lti_ratio <= 0.25:
            score += 0.35
        elif lti_ratio <= 0.45:
            score += 0.15
        else:
            score -= 0.30

        # Employment stability weighting
        if ey >= 5:
            score += 0.20
        elif ey >= 2:
            score += 0.10

        # Base noise for realistic variance
        prob = 1 / (1 + np.exp(- (score * 3.5 + np.random.normal(0, 0.35))))
        approved = 1 if prob >= 0.50 else 0
        loan_status.append(approved)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "loan_status": loan_status
    })

    return df

def save_dataset(filepath: str = "loans.csv", num_samples: int = 500, overwrite: bool = False):
    """
    Saves generated data to CSV file.
    If file already exists and overwrite is False, preserves existing data.
    """
    if os.path.exists(filepath) and not overwrite:
        print(f"'{filepath}' already exists. Skipping overwrite. Pass overwrite=True to recreate.")
        return

    df = generate_synthetic_data(num_samples=num_samples)
    df.to_csv(filepath, index=False)
    print(f"Successfully generated and saved {len(df)} records to '{filepath}'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Loan Applicant Dataset")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing loans.csv")
    args = parser.parse_args()

    save_dataset(filepath="loans.csv", num_samples=args.samples, overwrite=args.force)
