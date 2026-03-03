import numpy as np

# Load 3D MRI volumes
mean_vol = np.load("mean_vbm_3d.npy")      # shape: (91, 109, 91)
var_vol  = np.load("var_vbm_3d.npy")       # shape: (91, 109, 91)

X, Y, Z = mean_vol.shape
print("Volume shape:", (X, Y, Z))

# Create coordinates for every voxel in the grid
xs, ys, zs = np.meshgrid(
    np.arange(X), 
    np.arange(Y), 
    np.arange(Z), 
    indexing='ij'
)

coords = np.column_stack((
    xs.ravel(), 
    ys.ravel(), 
    zs.ravel()
))

intensity = mean_vol.ravel()
sigma = np.sqrt(var_vol.ravel()) + 1e-6  # avoid zero variance

print("Raw splat count:", coords.shape[0])

# === Prune low-intensity background voxels ===
threshold = np.percentile(intensity, 40)  # keep top 60% intensities
mask = intensity > threshold

coords = coords[mask]
intensity = intensity[mask]
sigma = sigma[mask]

print("After pruning:", coords.shape[0])

# === Construct Gaussian splat parameter array ===
# Format: [x, y, z, intensity, sigma]
gaussians = np.column_stack([coords, intensity, sigma])

np.save("gaussian_splats.npy", gaussians)
print("Saved gaussian_splats.npy with shape:", gaussians.shape)
