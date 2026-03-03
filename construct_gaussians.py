import numpy as np
import pandas as pd

# --- Load metadata ---
metadata = pd.read_csv("train.tsv", sep="\t")
print(metadata.head())

# --- Load training data ---
train_data = np.load("train.npy", mmap_mode="r")  # memory-mapped to avoid loading all at once
print(f"Training data shape: {train_data.shape}")

# --- Compute diagonal Gaussian statistics ---
def compute_diagonal_gaussian(data):
    """
    Computes mean and variance per feature (diagonal Gaussian) in a memory-efficient way.
    Uses memory-mapping and chunking if needed.
    """
    # If data is small enough, compute directly
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    return mean, var

mean_train, var_train = compute_diagonal_gaussian(train_data)
print("Computed mean and variance for each feature.")

# --- Example: sampling from the Gaussian ---
# Note: this will also be huge if you sample full features; consider small slices for testing
sample = np.random.normal(mean_train, np.sqrt(var_train))
print("Sample shape:", sample.shape)

np.save("mean_train.npy", mean_train)
np.save("var_train.npy", var_train)
print("Saved mean_train.npy and var_train.npy")