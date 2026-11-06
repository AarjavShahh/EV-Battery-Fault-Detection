import os
import numpy as np
from scipy.io import loadmat


DATA_PATH = os.path.join(
    "data",
    "battery_dataset",
    "B0005.mat"
)

print("=" * 60)
print("NASA B0005 SENSOR DATA STRUCTURE")
print("=" * 60)

mat_data = loadmat(DATA_PATH)

battery = mat_data["B0005"]

cycles = battery["cycle"][0, 0]

# Flatten 1 x 616 -> 616
cycles = cycles.reshape(-1)

print("Number of cycles:", len(cycles))

# ------------------------------------------------------------
# Inspect FIRST cycle
# ------------------------------------------------------------

cycle = cycles[0]

print("\nFirst cycle:")
print("Cycle type:", cycle["type"])
print("Ambient temperature:", cycle["ambient_temperature"])
print("Time:", cycle["time"])

# ------------------------------------------------------------
# Inspect data field
# ------------------------------------------------------------

cycle_data = cycle["data"]

print("\n" + "=" * 60)
print("CYCLE DATA")
print("=" * 60)

print("Type:", type(cycle_data))
print("Shape:", cycle_data.shape)
print("Dtype:", cycle_data.dtype)
print("Dtype names:", cycle_data.dtype.names)

# ------------------------------------------------------------
# Inspect each sensor field
# ------------------------------------------------------------

fields = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_charge",
    "Voltage_charge",
    "Time"
]

print("\n" + "=" * 60)
print("SENSOR FIELD STRUCTURE")
print("=" * 60)

for field in fields:

    value = cycle_data[field]

    print(f"\n{field}")
    print("  Type:", type(value))
    print("  Shape:", value.shape)
    print("  Dtype:", value.dtype)

    # Show a small sample
    print("  Sample:", value.flatten()[:5])

print("\n" + "=" * 60)