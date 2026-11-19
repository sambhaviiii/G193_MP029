
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Load datasets
train_df = pd.read_csv(DATA_PATH / "train_cleaned.csv")
later_df = pd.read_csv(DATA_PATH / "later_cleaned.csv")

# Sort later data by date
later_df["date"] = pd.to_datetime(later_df["date"])
later_df = later_df.sort_values("date").reset_index(drop=True)

# Split later data: first 70%, last 30%
split_index = int(len(later_df) * 0.70)

later_train = later_df.iloc[:split_index]
later_test = later_df.iloc[split_index:]

# Features and target
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

numeric_features = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg"
]

categorical_features = ["zone_type", "shift"]

def create_pipeline(regressor):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

# v1: Train on original cleaned training data only
X_v1 = train_df[FEATURES]
y_v1 = train_df["pick_seconds"]

# v2: Train on original training data + first 70% later data
v2_train = pd.concat(
    [train_df, later_train],
    ignore_index=True
)

X_v2 = v2_train[FEATURES]
y_v2 = v2_train["pick_seconds"]

# Common test data: last 30% of later data
X_test = later_test[FEATURES]
y_test = later_test["pick_seconds"]

# Create models
model_v1 = create_pipeline(
    RandomForestRegressor(n_estimators=100, random_state=42)
)

model_v2 = create_pipeline(
    GradientBoostingRegressor(random_state=42)
)

# Train both models
model_v1.fit(X_v1, y_v1)
model_v2.fit(X_v2, y_v2)

# Evaluate on the same test records
pred_v1 = model_v1.predict(X_test)
pred_v2 = model_v2.predict(X_test)

mae_v1 = mean_absolute_error(y_test, pred_v1)
mae_v2 = mean_absolute_error(y_test, pred_v2)

# Promotion decision: v2 must improve by at least 5%
improvement = ((mae_v1 - mae_v2) / mae_v1) * 100
promote_v2 = improvement >= 5

# Results
print("\n========== TIME-BASED MODEL COMPARISON ==========")
print("Original training rows:", len(train_df))
print("Later training rows:", len(later_train))
print("Final testing rows:", len(later_test))
print("Test period:", later_test["date"].min().date(),
      "to", later_test["date"].max().date())

print("\nV1 Random Forest MAE:", round(mae_v1, 2))
print("V2 Gradient Boosting MAE:", round(mae_v2, 2))
print("Improvement:", round(improvement, 2), "%")

if promote_v2:
    print("\nDECISION: PROMOTE V2")
else:
    print("\nDECISION: KEEP V1")
