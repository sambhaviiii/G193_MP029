
from pathlib import Path
import pandas as pd

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw"

# Load training data
train = pd.read_csv(DATA_PATH / "train.csv")

print("Original training data shape:", train.shape)
print(train.head())

# Remove duplicate rows
print("\nDuplicates before cleaning:", train.duplicated().sum())

train = train.drop_duplicates()

print("Shape after removing duplicates:", train.shape)
print("Duplicates after cleaning:", train.duplicated().sum())

# Correct spelling variants in zone_type
train["zone_type"] = (
    train["zone_type"]
    .str.strip()
    .str.lower()
)

zone_mapping = {
    "amb.": "ambient",
    "bul.": "bulky",
    "chi.": "chilled"
}

train["zone_type"] = train["zone_type"].replace(zone_mapping)

print("\nZone types after cleaning:")
print(train["zone_type"].value_counts())

print("Unique zone types:", train["zone_type"].nunique())

# Correct target values that are approximately 10 times too large
typo_mask = train["pick_seconds"] > 1000

print("\nSuspicious target values before correction:")
print(typo_mask.sum())

train.loc[typo_mask, "pick_seconds"] = (
    train.loc[typo_mask, "pick_seconds"] / 10
)

print("Suspicious target values after correction:")
print((train["pick_seconds"] > 1000).sum())

print("\nCorrected target values:")
print(train.loc[typo_mask, ["record_id", "pick_time_from_scanner", "pick_seconds"]].to_string(index=False))


# Remove data leakage column
train = train.drop(columns=["pick_time_from_scanner"])

print("\nColumns after removing leakage:")
print(train.columns.tolist())

print("\nFinal training data shape so far:", train.shape)


# Save cleaned dataset
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

output_file = PROCESSED_PATH / "train_cleaned.csv"

train.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved location:", output_file)
print("Final dataset shape:", train.shape)

# Load later dataset
later = pd.read_csv(DATA_PATH / "later.csv")

print("\n========== CLEAN LATER DATASET ==========")
print("Original later data shape:", later.shape)

# Remove duplicate rows
later = later.drop_duplicates()

# Normalize zone types
later["zone_type"] = (
    later["zone_type"]
    .str.strip()
    .str.lower()
)

later["zone_type"] = later["zone_type"].replace(zone_mapping)

# Remove leakage column
later = later.drop(columns=["pick_time_from_scanner"])

# Save cleaned later dataset
later.to_csv(PROCESSED_PATH / "later_cleaned.csv", index=False)

print("Cleaned later data shape:", later.shape)
print("Remaining duplicates:", later.duplicated().sum())
print("Zone types:", later["zone_type"].unique())
print("Saved: data/processed/later_cleaned.csv")
