import numpy as np
import os
from sklearn.cluster import KMeans

# ============================================
# CONFIG
# ============================================
VBM_DIR = "../VBM_extracted"      # folder with per-subject VBM .npy files
OUT_DIR = "../VBM_gaussians"      # folder to save Gaussian splats
os.makedirs(OUT_DIR, exist_ok=True)

K = 500  # Number of Gaussians per scan

# ============================================
# FUNCTION: Gaussian Splatting
# ============================================
def gaussian_splatting(volume, K):
    """
    Convert a 3D VBM volume into K Gaussian centers + weights.
    """
    # Select voxels consistently
    mask = volume > 0
    coords = np.array(np.nonzero(mask)).T
    weights = volume[mask].flatten()

    if len(coords) <= K:
        # Not enough voxels, just use them all
        return coords, weights

    # Weighted KMeans clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_

    # Compute cluster weights
    labels = kmeans.labels_
    cluster_weights = np.zeros(K)
    for i in range(K):
        cluster_weights[i] = weights[labels == i].sum()

    return centers, cluster_weights

# ============================================
# PROCESS ALL VBM SCANS
# ============================================
vbm_files = sorted(os.listdir(VBM_DIR))
print(f"Found {len(vbm_files)} VBM files.")

for i, fname in enumerate(vbm_files):
    path = os.path.join(VBM_DIR, fname)
    volume = np.load(path)
    
    centers, weights = gaussian_splatting(volume, K)
    
    out_path = os.path.join(OUT_DIR, fname.replace(".npy", "_gaussians.npz"))
    np.savez_compressed(out_path, centers=centers, weights=weights)
    
    if (i+1) % 50 == 0 or (i+1) == len(vbm_files):
        print(f"Processed {i+1}/{len(vbm_files)} scans")