"""
train_kmeans.py
Project: Customer Persona Segmenter (Unsupervised)
Purpose: Normalizes annual_income_k / spending_score with StandardScaler,
         trains a 3-cluster K-Means model, maps each centroid to a
         marketing persona, and exports the saved artifacts (.pkl) for
         main_unsupervised.py (FastAPI backend) to load.
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
DATA_PATH = "customers.csv"
df = pd.read_csv(DATA_PATH)

FEATURES = ["annual_income_k", "spending_score"]
X = df[FEATURES]

# ---------------------------------------------------------
# 2. Normalize features (K-Means is distance-based, scaling matters)
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. Train K-Means with k=3 (per project spec)
# ---------------------------------------------------------
K = 3
model = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)

score = silhouette_score(X_scaled, df["cluster"])
print(f"Trained K-Means with k={K}  |  silhouette score: {score:.4f}")

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

centers_original = scaler.inverse_transform(model.cluster_centers_)
print("\nCluster centers (original scale):")
for i, c in enumerate(centers_original):
    print(f"  Cluster {i}: annual_income_k={c[0]:.1f}, spending_score={c[1]:.1f}")


income_mean = df["annual_income_k"].mean()
spending_mean = df["spending_score"].mean()

def label_persona(income, spending):
    high_income = income >= income_mean
    high_spending = spending >= spending_mean
    if high_income and high_spending:
        return "Premium Spenders"
    if high_income and not high_spending:
        return "Careful High Earners"
    if not high_income and high_spending:
        return "Aspirational Shoppers"
    return "Budget Conscious"

persona_map = {
    i: label_persona(c[0], c[1]) for i, c in enumerate(centers_original)
}
df["persona"] = df["cluster"].map(persona_map)

print("\nCluster -> Persona mapping:")
for cluster_id, persona in persona_map.items():
    print(f"  Cluster {cluster_id}: {persona}")


with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("persona_map.pkl", "wb") as f:
    pickle.dump(persona_map, f)

df.to_csv("customers_with_clusters.csv", index=False)

print("\nSaved kmeans_model.pkl, scaler.pkl, persona_map.pkl, customers_with_clusters.csv")
