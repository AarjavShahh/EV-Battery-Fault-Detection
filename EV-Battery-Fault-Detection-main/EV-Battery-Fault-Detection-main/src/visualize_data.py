import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


print("=" * 60)
print("BATTERY DATA VISUALIZATION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

data_file = Path("data/battery_data.csv")

if not data_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_file}"
    )

df = pd.read_csv(data_file)

print(f"Dataset loaded: {df.shape}")
print()


# ---------------------------------------------------------
# 2. Create results folder
# ---------------------------------------------------------

results_folder = Path("results")

results_folder.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# 3. Plot voltage
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["voltage"],
    linewidth=0.7
)

plt.title(
    "NASA Battery - Voltage Measurement"
)

plt.xlabel("Measurement Sample")

plt.ylabel("Voltage (V)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_folder / "voltage_plot.png",
    dpi=150
)

plt.show()


# ---------------------------------------------------------
# 4. Plot current
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["current"],
    linewidth=0.7
)

plt.title(
    "NASA Battery - Current Measurement"
)

plt.xlabel("Measurement Sample")

plt.ylabel("Current (A)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_folder / "current_plot.png",
    dpi=150
)

plt.show()


# ---------------------------------------------------------
# 5. Plot temperature
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["temperature"],
    linewidth=0.7
)

plt.title(
    "NASA Battery - Temperature Measurement"
)

plt.xlabel("Measurement Sample")

plt.ylabel("Temperature (°C)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_folder / "temperature_plot.png",
    dpi=150
)

plt.show()


# ---------------------------------------------------------
# 6. Plot first 10 cycles
# ---------------------------------------------------------

first_cycles = df[
    df["cycle"] <= 10
]


plt.figure(figsize=(12, 5))

plt.plot(
    first_cycles["voltage"],
    linewidth=0.7
)

plt.title(
    "Battery Voltage - First 10 Cycles"
)

plt.xlabel("Measurement Sample")

plt.ylabel("Voltage (V)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_folder / "first_10_cycles_voltage.png",
    dpi=150
)

plt.show()


# ---------------------------------------------------------
# 7. Temperature vs voltage
# ---------------------------------------------------------

sample = df.sample(
    min(10000, len(df)),
    random_state=42
)


plt.figure(figsize=(8, 6))

plt.scatter(
    sample["voltage"],
    sample["temperature"],
    s=3,
    alpha=0.5
)

plt.title(
    "Battery Voltage vs Temperature"
)

plt.xlabel("Voltage (V)")

plt.ylabel("Temperature (°C)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_folder / "voltage_vs_temperature.png",
    dpi=150
)

plt.show()


# ---------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------

print("=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)

print()
print("Plots saved in:")

print(
    results_folder
)

print()
print("Created files:")

print("1. voltage_plot.png")
print("2. current_plot.png")
print("3. temperature_plot.png")
print("4. first_10_cycles_voltage.png")
print("5. voltage_vs_temperature.png")

print("=" * 60)