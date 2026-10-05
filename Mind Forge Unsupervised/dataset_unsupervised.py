import pandas as pd
import numpy as np

np.random.seed(42)

customers = pd.DataFrame({
    "annual_income_k": np.random.randint(20, 150, 600),
    "spending_score": np.random.randint(1, 100, 600)
})

customers.to_csv("customers.csv", index=False)

print("Customer dataset created successfully!")
print(customers.head())
print("\nDataset shape:", customers.shape)