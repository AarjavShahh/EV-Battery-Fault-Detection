import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path


print("=" * 60)
print("NASA BATTERY DATA EXTRACTION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Dataset path
# ---------------------------------------------------------

mat_file = Path("data/battery_dataset/B0005.mat")

if not mat_file.exists():
    raise FileNotFoundError(
        f"Dataset file not found: {mat_file}"
    )


# ---------------------------------------------------------
# 2. Load MATLAB file
# ---------------------------------------------------------

battery = loadmat(mat_file)

# B0005 is the top-level variable inside the .mat file
battery_data = battery["B0005"]

# Access the cycle structure
cycles = battery_data["cycle"][0, 0]

# Convert (1, 616) into a simple 1-D array
cycles = cycles.reshape(-1)

print(f"Number of cycles: {len(cycles)}")


# ---------------------------------------------------------
# 3. Extract sensor data
# ---------------------------------------------------------

rows = []

for cycle_number, cycle in enumerate(cycles, start=1):

    print(
        f"Processing cycle {cycle_number}/{len(cycles)}",
        end="\r"
    )

    try:

        # ---------------------------------------------
        # Cycle information
        # ---------------------------------------------

        cycle_type = str(
            np.asarray(cycle["type"]).squeeze()
        )

        ambient_temperature = float(
            np.asarray(
                cycle["ambient_temperature"]
            ).squeeze()
        )

        # ---------------------------------------------
        # Sensor data
        # ---------------------------------------------

        data = cycle["data"]

        voltage = np.asarray(
            data["Voltage_measured"][0, 0]
        ).reshape(-1)

        current = np.asarray(
            data["Current_measured"][0, 0]
        ).reshape(-1)

        temperature = np.asarray(
            data["Temperature_measured"][0, 0]
        ).reshape(-1)

        time = np.asarray(
            data["Time"][0, 0]
        ).reshape(-1)

        # ---------------------------------------------
        # Make sure all arrays have same length
        # ---------------------------------------------

        n = min(
            len(voltage),
            len(current),
            len(temperature),
            len(time)
        )

        # ---------------------------------------------
        # Store measurements
        # ---------------------------------------------

        for i in range(n):

            rows.append({
                "cycle": cycle_number,
                "cycle_type": cycle_type,
                "ambient_temperature": ambient_temperature,
                "time": time[i],
                "voltage": voltage[i],
                "current": current[i],
                "temperature": temperature[i]
            })

    except Exception as e:

        print()

        print(
            f"WARNING: Could not process cycle "
            f"{cycle_number}: {e}"
        )


# ---------------------------------------------------------
# 4. Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(rows)


# ---------------------------------------------------------
# 5. Clean the dataset
# ---------------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()


# ---------------------------------------------------------
# 6. Reset index
# ---------------------------------------------------------

df = df.reset_index(drop=True)


# ---------------------------------------------------------
# 7. Save CSV
# ---------------------------------------------------------

output_file = Path(
    "data/battery_data.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print()

print("=" * 60)
print("EXTRACTION COMPLETED")
print("=" * 60)

print(
    f"Total rows: {len(df):,}"
)

print()

print("Columns:")

print(
    df.columns.tolist()
)

print()

print("Dataset shape:")

print(
    df.shape
)

print()

print("First 5 rows:")

print(
    df.head()
)

print()

print("Missing values:")

print(
    df.isnull().sum()
)

print()

print("Basic statistics:")

print(
    df.describe()
)

print()

print("=" * 60)

print(
    f"CSV saved to: {output_file}"
)

print("=" * 60)