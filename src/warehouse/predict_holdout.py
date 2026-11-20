
"""
Holdout Predictions Pipeline
Author: Rohit Mukhopadhyay (S24CSEU0375) - Model Development & MLOps Lead
SCSE3040 MLOps Lucknow Fulfilment Centre - Warehouse Picking Time Prediction
- Generates predictions for holdout_features.csv using champion model
- Outputs predictions/holdout_predictions.csv (record_id, prediction)
"""

from pathlib import Path
import pandas as pd
import mlflow
import mlflow.pyfunc

# Project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load holdout data
holdout_path = PROJECT_ROOT / "data" / "raw" / "holdout_features.csv"
holdout_df = pd.read_csv(holdout_path)

# Required input features
FEATURES = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
    "zone_type",
    "shift"
]

# Prepare features
X_holdout = holdout_df[FEATURES].copy()

# Normalize zone types exactly as in training
X_holdout["zone_type"] = (
    X_holdout["zone_type"].str.strip().str.lower()
)

zone_mapping = {
    "amb.": "ambient",
    "bul.": "bulky",
    "chi.": "chilled"
}

X_holdout["zone_type"] = X_holdout["zone_type"].replace(zone_mapping)

# Load champion model from MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = "models:/warehouse_picking_model@champion"
model = mlflow.pyfunc.load_model(model_uri)

# Generate predictions
predictions = model.predict(X_holdout)

# Create required output
output_df = pd.DataFrame({
    "record_id": holdout_df["record_id"],
    "prediction": predictions
})

# Save predictions
output_path = PROJECT_ROOT / "predictions" / "holdout_predictions.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

output_df.to_csv(output_path, index=False)

# Verify output
print("\n========== HOLDOUT PREDICTIONS ==========")
print("Champion model:", model_uri)
print("Total predictions:", len(output_df))
print("Output file:", output_path)
print("\nFirst 5 predictions:")
print(output_df.head())

print("\nOutput columns:", list(output_df.columns))
print("Missing predictions:", output_df["prediction"].isna().sum())
