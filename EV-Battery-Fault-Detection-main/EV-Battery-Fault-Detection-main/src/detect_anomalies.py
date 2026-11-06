import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

print("=" * 60)
print("BATTERY ANOMALY DETECTION")
print("=" * 60)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
data_folder = Path("data/processed")
model_folder = Path("models")
results_folder = Path("results")

results_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load model and data
# ---------------------------------------------------------
print("\nLoading trained model...")

model = load_model(
    model_folder / "lstm_autoencoder.keras"
)

print("Model loaded successfully.")

print("\nLoading test sequences...")

X_test = np.load(
    data_folder / "X_test.npy"
)

print(f"Test data shape: {X_test.shape}")

# ---------------------------------------------------------
# Reconstruct test sequences
# ---------------------------------------------------------
print("\nReconstructing test sequences...")

X_pred = model.predict(
    X_test,
    batch_size=128,
    verbose=1
)

print("Reconstruction completed.")

# ---------------------------------------------------------
# Calculate reconstruction error
# ---------------------------------------------------------
print("\nCalculating reconstruction errors...")

# Mean squared error for every sequence
reconstruction_errors = np.mean(
    np.square(X_test - X_pred),
    axis=(1, 2)
)

print("\nReconstruction error statistics:")

print(
    pd.Series(reconstruction_errors).describe()
)

# ---------------------------------------------------------
# Determine anomaly threshold
# ---------------------------------------------------------
# We use the 95th percentile as the initial threshold.
threshold = np.percentile(
    reconstruction_errors,
    95
)

print("\nAnomaly threshold:")
print(f"{threshold:.8f}")

# ---------------------------------------------------------
# Detect anomalies
# ---------------------------------------------------------
anomalies = reconstruction_errors > threshold

normal_count = np.sum(~anomalies)
anomaly_count = np.sum(anomalies)

print("\nDetection results:")
print(f"Normal sequences : {normal_count}")
print(f"Anomalous sequences: {anomaly_count}")

print(
    f"Anomaly percentage: "
    f"{(anomaly_count / len(anomalies)) * 100:.2f}%"
)

# ---------------------------------------------------------
# Save reconstruction errors
# ---------------------------------------------------------
error_df = pd.DataFrame({
    "reconstruction_error": reconstruction_errors,
    "status": np.where(
        anomalies,
        "ANOMALY",
        "NORMAL"
    )
})

error_file = results_folder / "anomaly_results.csv"

error_df.to_csv(
    error_file,
    index=False
)

print(f"\nResults saved to: {error_file}")

# ---------------------------------------------------------
# Plot reconstruction error
# ---------------------------------------------------------
print("\nCreating anomaly detection plot...")

plt.figure(figsize=(12, 6))

plt.plot(
    reconstruction_errors,
    label="Reconstruction Error"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Sequence")
plt.ylabel("Reconstruction Error")
plt.title("LSTM Autoencoder Anomaly Detection")
plt.legend()
plt.grid(True)

plot_file = results_folder / "reconstruction_error.png"

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Plot saved to: {plot_file}")

# ---------------------------------------------------------
# Error distribution
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.hist(
    reconstruction_errors,
    bins=100
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Sequences")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.grid(True)

distribution_file = (
    results_folder /
    "reconstruction_error_distribution.png"
)

plt.savefig(
    distribution_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Distribution plot saved to: "
    f"{distribution_file}"
)

# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------
print()
print("=" * 60)
print("ANOMALY DETECTION COMPLETED")
print("=" * 60)

print(f"Threshold: {threshold:.8f}")
print(f"Normal sequences: {normal_count}")
print(f"Anomalous sequences: {anomaly_count}")
print("=" * 60)