
"""
Model Monitoring & Data Drift Analysis Module
Author: Rohit Mukhopadhyay (S24CSEU0375) - Model Development & MLOps Lead
SCSE3040 MLOps Lucknow Fulfilment Centre - Warehouse Picking Time Prediction
- 10-bin Population Stability Index (PSI) computation between train and later data
- Subgroup performance monitoring across shifts (Morning, Evening, Night)
"""

import pandas as pd
import numpy as np

# Load cleaned datasets
train = pd.read_csv("data/processed/train_cleaned.csv")
later = pd.read_csv("data/processed/later_cleaned.csv")

# Select numeric columns common to both datasets
numeric_columns = train.select_dtypes(include="number").columns
numeric_columns = [
    col for col in numeric_columns if col in later.columns
]

results = []

# Calculate PSI for each numeric column
for col in numeric_columns:
    train_values = train[col].dropna()
    later_values = later[col].dropna()

    # Create 10 bins using the training data
    bins = np.unique(
        np.quantile(train_values, np.linspace(0, 1, 11))
    )

    if len(bins) < 2:
        print(f"Skipping {col}: insufficient unique values")
        continue

    bins[0] = -np.inf
    bins[-1] = np.inf

    train_bins = pd.cut(train_values, bins=bins)
    later_bins = pd.cut(later_values, bins=bins)

    train_pct = train_bins.value_counts(
        normalize=True, sort=False
    )
    later_pct = later_bins.value_counts(
        normalize=True, sort=False
    )

    # Avoid division by zero
    epsilon = 0.0001
    train_pct = train_pct.clip(lower=epsilon)
    later_pct = later_pct.clip(lower=epsilon)

    # Normalize percentages
    train_pct = train_pct / train_pct.sum()
    later_pct = later_pct / later_pct.sum()

    psi = (
        (later_pct - train_pct)
        * np.log(later_pct / train_pct)
    ).sum()

    results.append({
        "column": col,
        "PSI": round(psi, 4)
    })

# Display results from highest to lowest drift
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="PSI", ascending=False
)

print("\n========== DATA DRIFT MONITORING ==========")
print(results_df.to_string(index=False))

# ==========================================
# MODEL PERFORMANCE MONITORING BY SHIFT
# ==========================================

import mlflow.pyfunc
from sklearn.metrics import mean_absolute_error

# Load the v1 Random Forest model
model = mlflow.pyfunc.load_model(
    "models:/warehouse_picking_model/1"
)

# Features used by the model
features = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
    "zone_type",
    "shift"
]

X = later[features]
y = later["pick_seconds"]

# Generate predictions
predictions = model.predict(X)

later["prediction"] = predictions

# Calculate MAE separately for each shift
print("\n========== MAE BY SHIFT ==========")

shift_results = []

for shift_name, group in later.groupby("shift"):

    mae = mean_absolute_error(
        group["pick_seconds"],
        group["prediction"]
    )

    shift_results.append({
        "shift": shift_name,
        "MAE": round(mae, 2)
    })

shift_results_df = pd.DataFrame(shift_results)

print(shift_results_df.to_string(index=False))

# Identify the worst-performing shift
worst_shift = shift_results_df.loc[
    shift_results_df["MAE"].idxmax()
]

print("\nWorst-performing shift:")
print(worst_shift.to_string())

