from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


# Find project folder
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Load datasets
train_df = pd.read_csv(DATA_PATH / "train_cleaned.csv")
later_df = pd.read_csv(DATA_PATH / "later_cleaned.csv")

# Sort later data by date
later_df["date"] = pd.to_datetime(later_df["date"])
later_df = later_df.sort_values("date").reset_index(drop=True)

# Use first 70% of later data for training
split_index = int(len(later_df) * 0.70)
later_train = later_df.iloc[:split_index]

# Combine training datasets
training_data = pd.concat(
    [train_df, later_train],
    ignore_index=True
)

# Features used by the API
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

NUMERIC_FEATURES = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg"
]

CATEGORICAL_FEATURES = ["zone_type", "shift"]

X_train = training_data[FEATURES]
y_train = training_data["pick_seconds"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        )
    ]
)

# Champion model: Gradient Boosting v2
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

# Train
model.fit(X_train, y_train)

# Save model
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Model trained and saved successfully!")
print(f"Training rows: {len(training_data)}")
print(f"Model location: {MODEL_PATH}")