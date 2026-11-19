
from pathlib import Path
import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset paths
DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Load the three datasets
train = pd.read_csv(DATA_PATH / "train.csv")
later = pd.read_csv(DATA_PATH / "later.csv")
holdout = pd.read_csv(DATA_PATH / "holdout_features.csv")

# Display basic information
print("\n========== DATASET SHAPES ==========")
print("Train:", train.shape)
print("Later:", later.shape)
print("Holdout:", holdout.shape)

print("\n========== TRAIN COLUMNS ==========")
print(train.columns.tolist())

print("\n========== TRAIN DATA TYPES ==========")
print(train.dtypes)

print("\n========== FIRST 5 TRAIN ROWS ==========")
print(train.head())

print("\n========== MISSING VALUES ==========")
print(train.isnull().sum())

print("\n========== DUPLICATE CHECK ==========")

print("Exact duplicate rows:", train.duplicated().sum())
print("Duplicate record IDs:", train["record_id"].duplicated().sum())

print("\nRows with duplicate record IDs:")
print(train[train["record_id"].duplicated(keep=False)].head(10))


print("\n========== DATA LEAKAGE CHECK ==========")

print("Train columns:", train.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())

print("\nColumns present in train but missing in holdout:")
print(set(train.columns) - set(holdout.columns))

print("\nCorrelation with target:")
print(train.corr(numeric_only=True)["pick_seconds"].sort_values(ascending=False))


print("\n========== LATER DATASET CHECK ==========")

print("Later exact duplicate rows:", later.duplicated().sum())
print("Later duplicate record IDs:", later["record_id"].duplicated().sum())

print("\n========== UNIT MIX-UP CHECK ==========")

numeric_columns = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
    "pick_seconds"
]

for column in numeric_columns:
    print(f"\n--- {column} by zone_type ---")
    print(train.groupby("zone_type")[column].agg(["min", "mean", "max"]))

    print(f"\n--- {column} by shift ---")
    print(train.groupby("shift")[column].agg(["min", "mean", "max"]))

print("\n========== TARGET TYPO CHECK ==========")

print("Target statistics:")
print(train["pick_seconds"].describe())

print("\nLargest target values:")
print(
    train[["record_id", "date", "pick_seconds"]]
    .sort_values("pick_seconds", ascending=False)
    .head(20)
)

print("\nTarget values greater than 1000:")
print(
    train[train["pick_seconds"] > 1000][
        ["record_id", "date", "pick_seconds"]
    ]
)

print("\n========== SENTINEL VALUE CHECK ==========")

for column in train.select_dtypes(include="number").columns:
    print(f"\n{column}")
    print(train[column].value_counts().head(15))


print("\n========== SUSPICIOUS SENTINEL VALUES ==========")

sentinel_values = [-999, -1, 0, 999, 9999]

for column in train.select_dtypes(include="number").columns:
    for value in sentinel_values:
        count = (train[column] == value).sum()

        if count > 0:
            print(f"{column}: value {value} appears {count} times")

print("\n========== STUCK SENSOR CHECK ==========")

train_sorted = train.sort_values("date").copy()

for column in train.select_dtypes(include="number").columns:
    print(f"\nChecking column: {column}")

    train_sorted["same_value_group"] = (
        train_sorted[column] != train_sorted[column].shift()
    ).cumsum()

    runs = (
        train_sorted.groupby("same_value_group")
        .agg(
            value=(column, "first"),
            count=(column, "size"),
            first_date=("date", "first"),
            last_date=("date", "last")
        )
        .reset_index(drop=True)
    )

    suspicious = runs[runs["count"] >= 10]

    if not suspicious.empty:
        print(suspicious.to_string(index=False))
print("\n========== VERIFY SHIFT HOUR STUCK READING ==========")

check = train_sorted[
    (train_sorted["shift_hour"] == 9) &
    (train_sorted["date"] >= "2026-03-29") &
    (train_sorted["date"] <= "2026-04-14")
]

print("Total records:", len(check))
print("\nRecords by date:")
print(check.groupby("date").size().to_string())

print("\nUnique shift hours during this period:")
print(check["shift_hour"].unique())

print("\n========== CHECK UNIT MIX-UP ==========")

for column in [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "order_weight_kg"
]:
    print(f"\n{column}")
    print("TRAIN:")
    print(train[column].describe())

    print("LATER:")
    print(later[column].describe())

print("\n========== SHELF HEIGHT UNIT INVESTIGATION ==========")

print("\nTRAIN shelf heights (first 15):")
print(train["shelf_height_m"].head(15).to_list())

print("\nLATER shelf heights (first 15):")
print(later["shelf_height_m"].head(15).to_list())

print("\nTRAIN shelf height range:")
print(train["shelf_height_m"].min(), "to", train["shelf_height_m"].max())

print("\nLATER shelf height range:")
print(later["shelf_height_m"].min(), "to", later["shelf_height_m"].max())

print("\nLATER values greater than 2.5:")
print(later[later["shelf_height_m"] > 2.5][
    ["record_id", "date", "shelf_height_m"]
].head(20).to_string(index=False))

print("\n========== TEST SHELF HEIGHT CONVERSION ==========")

for factor in [1, 1.2, 1.5, 2, 3, 10, 100]:
    converted_mean = later["shelf_height_m"].mean() / factor
    converted_min = later["shelf_height_m"].min() / factor
    converted_max = later["shelf_height_m"].max() / factor

    print(
        f"Divide by {factor}: "
        f"min={converted_min:.3f}, "
        f"mean={converted_mean:.3f}, "
        f"max={converted_max:.3f}"
    )

print("\nTRAIN reference:")
print(train["shelf_height_m"].describe())

print("\n========== VERIFY FACTOR 1.5 ==========")

print("TRAIN shelf height quantiles:")
print(train["shelf_height_m"].quantile([0, 0.25, 0.5, 0.75, 1]))

print("\nLATER shelf height quantiles before conversion:")
print(later["shelf_height_m"].quantile([0, 0.25, 0.5, 0.75, 1]))

print("\nLATER shelf height quantiles after dividing by 1.5:")
print((later["shelf_height_m"] / 1.5).quantile([0, 0.25, 0.5, 0.75, 1]))

print("\nFirst 10 converted later values:")
print((later["shelf_height_m"] / 1.5).head(10).round(3).to_list())

print("\n========== SHELF HEIGHT BY ZONE TYPE ==========")

print("\nTRAIN:")
print(train.groupby("zone_type")["shelf_height_m"].agg(["count", "min", "mean", "max"]).round(3))

print("\nLATER:")
print(later.groupby("zone_type")["shelf_height_m"].agg(["count", "min", "mean", "max"]).round(3))

print("\n========== SHELF HEIGHT BY SHIFT ==========")

print("\nTRAIN:")
print(train.groupby("shift")["shelf_height_m"].agg(["count", "min", "mean", "max"]).round(3))

print("\nLATER:")
print(later.groupby("shift")["shelf_height_m"].agg(["count", "min", "mean", "max"]).round(3))

print("\n========== CHECK TARGET VALUES AGAINST SCANNER ==========")

large_targets = train[train["pick_seconds"] > 1000][
    ["record_id", "date", "pick_time_from_scanner", "pick_seconds"]
].copy()

large_targets["target_scanner_ratio"] = (
    large_targets["pick_seconds"] / large_targets["pick_time_from_scanner"]
)

print(large_targets.to_string(index=False))

print("\n========== TARGET TYPO VERIFICATION ==========")

large_targets = train[train["pick_seconds"] > 1000].copy()

large_targets["corrected_target"] = (
    large_targets["pick_seconds"] / 10
)

large_targets["difference_from_scanner"] = (
    large_targets["corrected_target"]
    - large_targets["pick_time_from_scanner"]
)

print(
    large_targets[
        [
            "record_id",
            "pick_time_from_scanner",
            "pick_seconds",
            "corrected_target",
            "difference_from_scanner"
        ]
    ].to_string(index=False)
)

print("\n========== VERIFY POSSIBLE DATA LEAKAGE ==========")

corrected = train["pick_seconds"].copy()

corrected.loc[corrected > 1000] = (
    corrected.loc[corrected > 1000] / 10
)

difference = (
    corrected - train["pick_time_from_scanner"]
).abs()

print("Correlation after correcting target typos:")
print(corrected.corr(train["pick_time_from_scanner"]))

print("\nDifference between corrected target and scanner:")
print(difference.describe())

print("\nRows with difference <= 5 seconds:")
print((difference <= 5).sum(), "out of", len(train))

print("\nRows with difference > 5 seconds:")
print((difference > 5).sum())

print("\n========== VERIFY SPELLING VARIANTS ==========")

print("TRAIN zone_type values:")
print(train["zone_type"].value_counts().to_string())

print("\nLATER zone_type values:")
print(later["zone_type"].value_counts().to_string())

print("\nUnique zone types in TRAIN:", train["zone_type"].nunique())
print("Unique zone types in LATER:", later["zone_type"].nunique())

print("\n========== TEST ZONE TYPE NORMALIZATION ==========")

zone_mapping = {
    "amb.": "ambient",
    "bul.": "bulky",
    "chi.": "chilled"
}

normalized_zone = (
    train["zone_type"]
    .str.strip()
    .str.lower()
    .replace(zone_mapping)
)

print("Unique values after normalization:")
print(normalized_zone.value_counts())

print("\nUnique normalized categories:", normalized_zone.nunique())
print("Expected categories: 3")


print("\n========== VERIFY SENTINEL VALUES ==========")

for column in train.select_dtypes(include="number").columns:
    print(f"\nColumn: {column}")
    print(train[column].value_counts().head(10).to_string())

print("\n========== ZERO VALUE DETAILS ==========")

print("\nPicker experience = 0:")
print(train[train["picker_exp_months"] == 0][
    ["record_id", "date", "picker_exp_months", "pick_seconds"]
].to_string(index=False))

print("\nShift hour = 0:")
print(train[train["shift_hour"] == 0][
    ["record_id", "date", "shift_hour", "shift"]
].to_string(index=False))

print("\n========== NUMERIC RANGE VALIDATION ==========")

for column in train.select_dtypes(include="number").columns:
    print(
        f"{column}: "
        f"min={train[column].min()}, "
        f"max={train[column].max()}"
    )
