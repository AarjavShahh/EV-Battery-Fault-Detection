import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 60)
print("ANALYZING BATTERY ANOMALIES BY CYCLE")
print("=" * 60)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
data_file = Path("data/battery_data.csv")
results_folder = Path("results")

# ---------------------------------------------------------
# Load original dataset
# ---------------------------------------------------------
print("\nLoading original battery data...")

df = pd.read_csv(data_file)

print(f"Dataset shape: {df.shape}")

# ---------------------------------------------------------
# Load anomaly results
# ---------------------------------------------------------
print("\nLoading anomaly results...")

results_file = results_folder / "anomaly_results.csv"

results = pd.read_csv(results_file)

print(f"Anomaly result rows: {len(results)}")

# ---------------------------------------------------------
# Recreate test-cycle sequence mapping
# ---------------------------------------------------------
# The sequence creation script used:
# cycles 1-100  -> training
# cycles 101+   -> testing
#
# Each sequence was created inside one cycle.
# ---------------------------------------------------------

test_cycles = df["cycle"].unique()[100:]

sequence_records = []

sequence_length = 50

sequence_index = 0

for cycle_number in test_cycles:

    cycle_data = df[
        df["cycle"] == cycle_number
    ]

    number_of_points = len(cycle_data)

    if number_of_points < sequence_length:
        continue

    number_of_sequences = (
        number_of_points -
        sequence_length +
        1
    )

    for i in range(number_of_sequences):

        sequence_records.append({
            "sequence": sequence_index,
            "cycle": cycle_number
        })

        sequence_index += 1

# ---------------------------------------------------------
# Create mapping dataframe
# ---------------------------------------------------------
mapping = pd.DataFrame(sequence_records)

print(
    f"Generated sequence mapping: "
    f"{len(mapping)} sequences"
)

# ---------------------------------------------------------
# Check lengths
# ---------------------------------------------------------
if len(mapping) != len(results):

    print("\nWARNING:")
    print(
        "Sequence mapping length does not match "
        "anomaly results."
    )

    print(f"Mapping: {len(mapping)}")
    print(f"Results: {len(results)}")

    raise ValueError(
        "Sequence mapping mismatch."
    )

# ---------------------------------------------------------
# Combine mapping and anomaly results
# ---------------------------------------------------------
results["sequence"] = np.arange(
    len(results)
)

combined = mapping.merge(
    results,
    on="sequence"
)

# ---------------------------------------------------------
# Cycle-level anomaly statistics
# ---------------------------------------------------------
cycle_summary = (
    combined
    .groupby("cycle")
    .agg(
        total_sequences=(
            "sequence",
            "count"
        ),
        anomalous_sequences=(
            "status",
            lambda x: (x == "ANOMALY").sum()
        ),
        average_error=(
            "reconstruction_error",
            "mean"
        ),
        maximum_error=(
            "reconstruction_error",
            "max"
        )
    )
    .reset_index()
)

# ---------------------------------------------------------
# Calculate anomaly percentage
# ---------------------------------------------------------
cycle_summary["anomaly_percentage"] = (
    cycle_summary["anomalous_sequences"]
    / cycle_summary["total_sequences"]
    * 100
)

# ---------------------------------------------------------
# Sort most anomalous cycles first
# ---------------------------------------------------------
cycle_summary = cycle_summary.sort_values(
    "anomaly_percentage",
    ascending=False
)

# ---------------------------------------------------------
# Save cycle report
# ---------------------------------------------------------
output_file = (
    results_folder /
    "cycle_anomaly_summary.csv"
)

cycle_summary.to_csv(
    output_file,
    index=False
)

print(
    f"\nCycle summary saved to: "
    f"{output_file}"
)

# ---------------------------------------------------------
# Display top anomalous cycles
# ---------------------------------------------------------
print("\nTop 20 anomalous battery cycles:")

print(
    cycle_summary.head(20).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------
print("\nOverall statistics:")

print(
    f"Total test cycles: "
    f"{len(cycle_summary)}"
)

print(
    f"Cycles containing anomalies: "
    f"{(cycle_summary['anomalous_sequences'] > 0).sum()}"
)

print(
    f"Most anomalous cycle: "
    f"{cycle_summary.iloc[0]['cycle']}"
)

print(
    f"Highest anomaly percentage: "
    f"{cycle_summary.iloc[0]['anomaly_percentage']:.2f}%"
)

print()
print("=" * 60)
print("CYCLE ANOMALY ANALYSIS COMPLETED")
print("=" * 60)