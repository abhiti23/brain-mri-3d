import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load mean and variance (precomputed from training data)
# ---------------------------------------------------------
mean_train = np.load("mean_train.npy")
var_train = np.load("var_train.npy")
print("Loaded mean_train and var_train!")

# ---------------------------------------------------------
# 2. Define function for memory-safe z-score computation
# ---------------------------------------------------------
def compute_z_scores(data, mean, var, chunk_size=50):
    n_samples, n_features = data.shape
    z_scores = np.zeros((n_samples, n_features), dtype=np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = data[start:end]

        # z = (x − μ) / σ
        z_chunk = (chunk - mean) / np.sqrt(var + 1e-8)

        z_scores[start:end] = z_chunk
        print(f"Processed samples {start} → {end}")

    return z_scores

# ---------------------------------------------------------
# 3. Load metadata + test datasets (memory-mapped)
# ---------------------------------------------------------
internal_test_file = "internal_test.npy"
external_test_file = "external_test.npy"

internal_meta_file = "internal_test.tsv"
external_meta_file = "external_test.tsv"

internal_meta = pd.read_csv(internal_meta_file, sep="\t")
external_meta = pd.read_csv(external_meta_file, sep="\t")

internal_test = np.load(internal_test_file, mmap_mode='r')
external_test = np.load(external_test_file, mmap_mode='r')

print(f"Internal test shape: {internal_test.shape}")
print(f"External test shape: {external_test.shape}")

# ---------------------------------------------------------
# 4. Compute z-scores
# ---------------------------------------------------------
internal_z = compute_z_scores(internal_test, mean_train, var_train)
external_z = compute_z_scores(external_test, mean_train, var_train)

print("Z-scores computed!")

# ---------------------------------------------------------
# 5. Visualize voxel distributions
# ---------------------------------------------------------
example_voxels = [0, 1000, 50000, 200000, 1000000]

for v in example_voxels:
    plt.hist(internal_z[:, v], bins=50, alpha=0.7)
    plt.xlabel("Z-score")
    plt.ylabel("Frequency")
    plt.title(f"Internal Test: distribution of voxel {v}")
    plt.show()
