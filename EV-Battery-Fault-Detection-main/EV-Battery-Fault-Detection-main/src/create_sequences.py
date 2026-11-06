import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("CREATING LSTM SEQUENCES")
print("=" * 60)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
input_file = Path("data/battery_data.csv")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("\nLoading battery dataset...")

df = pd.read_csv(input_file)

print(f"Dataset shape: {df.shape}")

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------
features = [
    "voltage",
    "current",
    "temperature"
]

# ---------------------------------------------------------
# Select training and testing cycles
# ---------------------------------------------------------
# Early cycles are treated as normal/healthy behavior.
# Later cycles are used for testing anomaly detection.

train_cycles = df["cycle"].unique()[:100]
test_cycles = df["cycle"].unique()[100:]

print(f"\nTraining cycles: {train_cycles[0]} to {train_cycles[-1]}")
print(f"Testing cycles: {test_cycles[0]} to {test_cycles[-1]}")

train_df = df[df["cycle"].isin(train_cycles)].copy()
test_df = df[df["cycle"].isin(test_cycles)].copy()

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# ---------------------------------------------------------
# Normalize using TRAINING data only
# ---------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()

train_df[features] = scaler.fit_transform(
    train_df[features]
)

test_df[features] = scaler.transform(
    test_df[features]
)

# Save scaler
scaler_file = output_folder / "scaler.pkl"
joblib.dump(scaler, scaler_file)

print(f"\nScaler saved to: {scaler_file}")

# ---------------------------------------------------------
# Sequence creation function
# ---------------------------------------------------------
def create_sequences(dataframe, sequence_length=50):

    sequences = []

    # Process each battery cycle separately
    for cycle_number, cycle_data in dataframe.groupby("cycle"):

        values = cycle_data[features].values

        # Skip cycles that are too short
        if len(values) < sequence_length:
            continue

        # Sliding window
        for i in range(len(values) - sequence_length + 1):

            sequence = values[
                i:i + sequence_length
            ]

            sequences.append(sequence)

    return np.array(sequences, dtype=np.float32)


# ---------------------------------------------------------
# Create sequences
# ---------------------------------------------------------
SEQUENCE_LENGTH = 50

print("\nCreating training sequences...")

X_train = create_sequences(
    train_df,
    SEQUENCE_LENGTH
)

print(f"Training sequences shape: {X_train.shape}")

print("\nCreating testing sequences...")

X_test = create_sequences(
    test_df,
    SEQUENCE_LENGTH
)

print(f"Testing sequences shape: {X_test.shape}")

# ---------------------------------------------------------
# Save sequences
# ---------------------------------------------------------
train_file = output_folder / "X_train.npy"
test_file = output_folder / "X_test.npy"

np.save(train_file, X_train)
np.save(test_file, X_test)

print("\nFiles saved:")
print(f"Training: {train_file}")
print(f"Testing : {test_file}")

# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------
print("\nSequence information:")
print(f"Sequence length : {SEQUENCE_LENGTH}")
print(f"Number of features: {len(features)}")

print("\nTraining shape:")
print(X_train.shape)

print("\nTesting shape:")
print(X_test.shape)

print()
print("=" * 60)
print("SEQUENCE CREATION COMPLETED")
print("=" * 60)