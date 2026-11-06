import pandas as pd
from pathlib import Path

print("=" * 60)
print("CREATING LIGHTWEIGHT DASHBOARD DATA")
print("=" * 60)

ROOT = Path(__file__).resolve().parent.parent

input_file = ROOT / "data" / "battery_data.csv"
output_file = ROOT / "data" / "dashboard_data.csv"

print("\nLoading original dataset...")

df = pd.read_csv(input_file)

print(f"Original rows: {len(df):,}")

# ---------------------------------------------------------
# Create one row per battery cycle
# ---------------------------------------------------------

dashboard = (
    df.groupby("cycle")
    .agg(
        voltage=("voltage", "mean"),
        current=("current", "mean"),
        temperature=("temperature", "mean"),
        readings=("voltage", "count")
    )
    .reset_index()
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

dashboard.to_csv(
    output_file,
    index=False
)

print(f"\nDashboard rows: {len(dashboard):,}")
print(f"Saved to: {output_file}")

print()
print("=" * 60)
print("DASHBOARD DATA CREATED")
print("=" * 60)