import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

print("=" * 60)
print("LSTM AUTOENCODER TRAINING")
print("=" * 60)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
data_folder = Path("data/processed")
model_folder = Path("models")
model_folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load sequences
# ---------------------------------------------------------
print("\nLoading training and testing sequences...")

X_train = np.load(data_folder / "X_train.npy")
X_test = np.load(data_folder / "X_test.npy")

print(f"Training shape: {X_train.shape}")
print(f"Testing shape : {X_test.shape}")

# ---------------------------------------------------------
# Model parameters
# ---------------------------------------------------------
sequence_length = X_train.shape[1]
num_features = X_train.shape[2]

print("\nModel configuration:")
print(f"Sequence length : {sequence_length}")
print(f"Number features : {num_features}")

# ---------------------------------------------------------
# Build LSTM Autoencoder
# ---------------------------------------------------------
print("\nBuilding LSTM Autoencoder...")

inputs = layers.Input(
    shape=(sequence_length, num_features)
)

# Encoder
encoded = layers.LSTM(
    64,
    activation="tanh",
    return_sequences=False
)(inputs)

# Bottleneck
bottleneck = layers.RepeatVector(
    sequence_length
)(encoded)

# Decoder
decoded = layers.LSTM(
    64,
    activation="tanh",
    return_sequences=True
)(bottleneck)

outputs = layers.TimeDistributed(
    layers.Dense(num_features)
)(decoded)

model = Model(
    inputs,
    outputs
)

# ---------------------------------------------------------
# Compile
# ---------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="mse"
)

print("\nModel summary:")
model.summary()

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
print("\nStarting training...")

history = model.fit(
    X_train,
    X_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    shuffle=True
)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
model_file = model_folder / "lstm_autoencoder.keras"

model.save(model_file)

print(f"\nModel saved to: {model_file}")

# ---------------------------------------------------------
# Save training history
# ---------------------------------------------------------
history_file = model_folder / "training_history.npy"

np.save(
    history_file,
    history.history
)

print(f"Training history saved to: {history_file}")

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
print("\nEvaluating model on test data...")

test_loss = model.evaluate(
    X_test,
    X_test,
    verbose=1
)

print(f"\nTest reconstruction loss: {test_loss:.6f}")

print()
print("=" * 60)
print("LSTM AUTOENCODER TRAINING COMPLETED")
print("=" * 60)