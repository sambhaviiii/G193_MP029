
from pathlib import Path
import pandas as pd
import hashlib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Dataset fingerprint
original_data_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
data_sha256 = hashlib.sha256(
    original_data_path.read_bytes()
).hexdigest()

# Load data
train_df = pd.read_csv(DATA_PATH / "train_cleaned.csv")
later_df = pd.read_csv(DATA_PATH / "later_cleaned.csv")

later_df["date"] = pd.to_datetime(later_df["date"])
later_df = later_df.sort_values("date").reset_index(drop=True)

# Time-based split
split_index = int(len(later_df) * 0.70)
later_train = later_df.iloc[:split_index]
later_test = later_df.iloc[split_index:]

# Features
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

# Prepare training and testing data
v2_train = pd.concat(
    [train_df, later_train],
    ignore_index=True
)

X_v1 = train_df[FEATURES]
y_v1 = train_df["pick_seconds"]

X_v2 = v2_train[FEATURES]
y_v2 = v2_train["pick_seconds"]

X_test = later_test[FEATURES]
y_test = later_test["pick_seconds"]

# Create models
models = {
    "random_forest_v1": (
        create_pipeline(
            RandomForestRegressor(n_estimators=100, random_state=42)
        ),
        X_v1,
        y_v1
    ),
    "gradient_boosting_v2": (
        create_pipeline(
            GradientBoostingRegressor(random_state=42)
        ),
        X_v2,
        y_v2
    )
}

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MP029_Warehouse_Picking_Time")

MODEL_NAME = "warehouse_picking_model"
client = MlflowClient()

# Train, evaluate, and register both models
registered_run_ids = {}

for run_name, (model, X_train, y_train) in models.items():

    with mlflow.start_run(run_name=run_name) as run:

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        mlflow.log_param("model_type", run_name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            model,
            name="warehouse_picking_model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle"
        )

        registered_run_ids[run_name] = run.info.run_id

        print(run_name, "MAE:", round(mae, 2))
        print("Registered run:", run.info.run_id)

# Find registered model versions
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

version_by_run = {
    version.run_id: version.version
    for version in versions
}

v1_version = version_by_run[registered_run_ids["random_forest_v1"]]
v2_version = version_by_run[registered_run_ids["gradient_boosting_v2"]]

# Assign champion alias to v2
client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    v2_version
)

print("\n========== MODEL REGISTRY ==========")
print("Registered model:", MODEL_NAME)
print("V1 version:", v1_version)
print("V2 version:", v2_version)
print("Champion alias assigned to version:", v2_version)
