import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColumns and types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# TotalCharges has hidden blanks (" ") that pandas reads as text
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nHidden missing TotalCharges:", df["TotalCharges"].isnull().sum())

print("\nChurn distribution:\n", df["Churn"].value_counts())
print("\nChurn %:\n", df["Churn"].value_counts(normalize=True) * 100)

df["Churn"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Churned vs Non-churned customers")
plt.tight_layout()
plt.savefig("churn_distribution.png")
print("\nSaved churn_distribution.png")