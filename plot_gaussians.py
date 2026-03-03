import numpy as np
import matplotlib.pyplot as plt

# Load splats
splats = np.load("gaussian_splats.npy")
print(f"Splat array shape: {splats.shape}")

# Extract coordinates and intensity
x, y, z, intensity, sigma = splats.T

# Optional: normalize intensity for coloring
colors = (intensity - intensity.min()) / (intensity.max() - intensity.min())

# For plotting, we can subsample to speed things up
sample_idx = np.random.choice(len(x), size=50000, replace=False)
x_s, y_s, z_s, c_s = x[sample_idx], y[sample_idx], z[sample_idx], colors[sample_idx]

# Plot 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x_s, y_s, z_s, c=c_s, cmap='viridis', s=1, alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Gaussian Splat Visualization (subsampled)")
fig.colorbar(p, ax=ax, label='Normalized intensity')
plt.show()
