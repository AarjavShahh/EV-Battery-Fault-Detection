import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib


print("=" * 60)
print("BATTERY DATA PREPROCESSING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

input_file = Path("data/battery_data.csv")
output_folder = Path("data/processed")

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print(f"Original shape: {df.shape}")


# ---------------------------------------------------------
# 3. Select required features
# ---------------------------------------------------------

features = [
    "voltage",
    "current",
    "temperature"
]

data = df[features].copy()


# ---------------------------------------------------------
# 4. Remove invalid values
# ---------------------------------------------------------

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna()

print(
    f"Shape after cleaning: {data.shape}"
)


# ---------------------------------------------------------
# 5. Normalize data
# ---------------------------------------------------------

print("\nNormalizing sensor values...")

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(data)


# ---------------------------------------------------------
# 6. Save scaler
# ---------------------------------------------------------

scaler_file = output_folder / "scaler.pkl"

joblib.dump(
    scaler,
    scaler_file
)

print(
    f"Scaler saved to: {scaler_file}"
)


# ---------------------------------------------------------
# 7. Save normalized data
# ---------------------------------------------------------

normalized_df = pd.DataFrame(
    normalized_data,
    columns=features
)

normalized_file = (
    output_folder / "normalized_data.csv"
)

normalized_df.to_csv(
    normalized_file,
    index=False
)


# ---------------------------------------------------------
# 8. Display statistics
# ---------------------------------------------------------

print("\nOriginal data statistics:")

print(data.describe())


print("\nNormalized data statistics:")

print(normalized_df.describe())


# ---------------------------------------------------------
# 9. Final information
# ---------------------------------------------------------

print()
print("=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(
    f"Features used: {features}"
)

print(
    f"Normalized data shape: {normalized_df.shape}"
)

print(
    f"Normalized data saved to: {normalized_file}"
)

print(
    f"Scaler saved to: {scaler_file}"
)

print("=" * 60)