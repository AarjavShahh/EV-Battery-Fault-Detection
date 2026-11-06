from scipy.io import loadmat

mat_file = "data/battery_dataset/B0005.mat"

data = loadmat(mat_file)

print("=" * 60)
print("MAT FILE INSPECTION")
print("=" * 60)

print("\nTop-level variables:")
print(data.keys())

print("\nVariable details:")

for key, value in data.items():
    if not key.startswith("__"):
        print(f"\nName: {key}")
        print(f"Type: {type(value)}")
        print(f"Shape: {getattr(value, 'shape', 'N/A')}")
        print(f"Dtype: {getattr(value, 'dtype', 'N/A')}")