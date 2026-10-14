"""
Task 3: Real-World Dataset Analysis Project
Dataset: scikit-learn diabetes dataset (Efron et al.; one-year disease progression target).
The raw CSV in this project has intentionally introduced missing cells and duplicate rows
for demonstrating data-cleaning methods. Do not interpret this as the untouched original file.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / "diabetes_raw.csv")
print("Raw shape:", raw.shape)
print("\\nMissing values by column:\\n", raw.isna().sum())
print("\\nDuplicate observations:", raw.drop(columns=["record_id"]).duplicated().sum())

feature_cols = [c for c in raw.columns if c not in ("record_id", "target")]
data = raw.drop_duplicates(subset=feature_cols + ["target"]).copy()
for col in feature_cols + ["target"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
for col in feature_cols:
    data[col] = data[col].fillna(data[col].median())
data["target"] = data["target"].fillna(data["target"].median())
data = data.reset_index(drop=True)
data["record_id"] = np.arange(1, len(data) + 1)
data.to_csv(ROOT / "diabetes_cleaned_recreated.csv", index=False)

print("\\nCleaned shape:", data.shape)
print("\\nDescriptive statistics:\\n", data[feature_cols + ["target"]].describe().round(3))
print("\\nCorrelation with target:\\n", data[feature_cols + ["target"]].corr()["target"].sort_values(ascending=False))

(ROOT / "outputs").mkdir(exist_ok=True)
data[feature_cols + ["target"]].describe().T.to_csv(ROOT / "outputs" / "recreated_descriptive_statistics.csv")
data[feature_cols + ["target"]].corr().to_csv(ROOT / "outputs" / "recreated_correlation_matrix.csv")

plt.figure(figsize=(8, 5))
plt.hist(data["target"], bins=24, edgecolor="black")
plt.title("Distribution of One-Year Disease Progression Measure")
plt.xlabel("Target value (dataset scale)")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig(ROOT / "plots" / "recreated_target_distribution.png", dpi=160)
plt.close()
