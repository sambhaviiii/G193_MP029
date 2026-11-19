
from pathlib import Path
import pandas as pd
import hashlib
# Find project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load cleaned training data
DATA_PATH = PROJECT_ROOT / "data" / "processed"
df = pd.read_csv(DATA_PATH / "train_cleaned.csv")
# Calculate original training dataset fingerprint
original_data_path = PROJECT_ROOT / "data" / "raw" / "train.csv"

data_sha256 = hashlib.sha256(
    original_data_path.read_bytes()
).hexdigest()
# Select input features
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

# Separate features and target
X = df[FEATURES]
y = df["pick_seconds"]

print("Training data loaded successfully!")
print("Input features shape:", X.shape)
print("Target shape:", y.shape)
print("\nInput features:")
print(X.head())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

# Separate numeric and categorical columns
numeric_features = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg"
]

categorical_features = ["zone_type", "shift"]

# Preprocess the input data
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MP029_Warehouse_Picking_Time")

# Train and track the model
with mlflow.start_run(run_name="random_forest_v1"):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("data_sha256", data_sha256)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(
    model,
    name="warehouse_picking_model",
    serialization_format="cloudpickle"
    )

    print("\n========== MODEL RESULTS ==========")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Mean Absolute Error (MAE):", round(mae, 2))
