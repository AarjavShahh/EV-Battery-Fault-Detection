import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("CREATING FINAL PROJECT GRAPHS")
print("=" * 60)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
data_file = Path("data/battery_data.csv")
results_folder = Path("results")

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("\nLoading data...")

df = pd.read_csv(data_file)

anomaly_results = pd.read_csv(
    results_folder / "anomaly_results.csv"
)

cycle_summary = pd.read_csv(
    results_folder / "cycle_anomaly_summary.csv"
)

# ---------------------------------------------------------
# 1. Reconstruction error
# ---------------------------------------------------------
print("\nCreating reconstruction error graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    anomaly_results["reconstruction_error"]
)

threshold = anomaly_results[
    "reconstruction_error"
].quantile(0.95)

plt.axhline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Sequence")
plt.ylabel("Reconstruction Error")
plt.title(
    "LSTM Autoencoder Reconstruction Error"
)

plt.legend()
plt.grid(True)

plt.savefig(
    results_folder / "final_reconstruction_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# 2. Anomaly percentage by cycle
# ---------------------------------------------------------
print("Creating anomaly percentage graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    cycle_summary["cycle"],
    cycle_summary["anomaly_percentage"]
)

plt.xlabel("Battery Cycle")
plt.ylabel("Anomaly Percentage (%)")
plt.title(
    "Anomaly Percentage Across Battery Cycles"
)

plt.grid(True)

plt.savefig(
    results_folder / "anomaly_percentage_by_cycle.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# 3. Average reconstruction error by cycle
# ---------------------------------------------------------
print("Creating average error graph...")

cycle_sorted = cycle_summary.sort_values(
    "cycle"
)

plt.figure(figsize=(12, 6))

plt.plot(
    cycle_sorted["cycle"],
    cycle_sorted["average_error"]
)

plt.xlabel("Battery Cycle")
plt.ylabel("Average Reconstruction Error")
plt.title(
    "Average Reconstruction Error by Battery Cycle"
)

plt.grid(True)

plt.savefig(
    results_folder / "average_error_by_cycle.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# 4. Temperature by cycle
# ---------------------------------------------------------
print("Creating temperature graph...")

temperature_by_cycle = (
    df.groupby("cycle")["temperature"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    temperature_by_cycle["cycle"],
    temperature_by_cycle["temperature"]
)

plt.xlabel("Battery Cycle")
plt.ylabel("Average Temperature")
plt.title(
    "Battery Temperature Across Cycles"
)

plt.grid(True)

plt.savefig(
    results_folder / "temperature_by_cycle.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Completed
# ---------------------------------------------------------
print()
print("=" * 60)
print("FINAL GRAPHS CREATED")
print("=" * 60)

print("\nFiles created:")

print("1. final_reconstruction_error.png")
print("2. anomaly_percentage_by_cycle.png")
print("3. average_error_by_cycle.png")
print("4. temperature_by_cycle.png")

print()
print("All graphs are inside the results folder.")

print("=" * 60)