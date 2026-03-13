# this file visualizes how well the 3D images are approximated by a Tensor
# basis expansion. This file produces 2d image slices for a random subject
# and compares it to the corresponding image slice obtained from basis
# expansion.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skfda.representation.basis import BSplineBasis, TensorBasis
from skfda import FDataBasis

#############################
### LOAD a random image #####
L = 1500
random_subj = np.random.randint(0, L-1)
print(f"Visualization for subject {random_subj}")
file_name = f"data/raw/public_data_challenge/VBM_extracted/VBM_{random_subj}.npy"
rand_image = np.load(file_name)

total_slices = rand_image.shape[0]
n_slices = 5 # number of slices we want to visualize
indices = np.arange(0, total_slices, step=total_slices//n_slices)

#############################
### LOAD corresponding coefficients of the TensorBasis
file_name = f"artifacts/checkpoint.npz"
all_coefficients = np.load(file_name)["coefficients"]
all_coefficients = all_coefficients.squeeze(1)
coefficients = all_coefficients[random_subj, :]

#############################
### convert basis representation to grid representation ###
n_basis_per_dim = 8  # tune this to control compression ratio
bspline_univariate = BSplineBasis(domain_range=(0, 1),
                                             n_basis=n_basis_per_dim,
                                             order=4) # order 4 -> cubic spline
basis = TensorBasis([bspline_univariate, bspline_univariate, bspline_univariate])

# --- Your grid points (adjust ranges to match your original domain) ---
x_shape, y_shape, z_shape = rand_image.shape
grid_x = np.linspace(0, 1, x_shape)
grid_y = np.linspace(0, 1, y_shape)
grid_z = np.linspace(0, 1, z_shape)

# --- Reconstruct FDataBasis from coefficients ---
fd_basis = FDataBasis(basis=basis, coefficients=coefficients)
# --- Evaluate over the 3D grid ---
# FDataBasis.evaluate() expects points of shape (n_points, n_dim)
# So we need to build the full meshgrid and flatten it
xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
grid_points_flat = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])  # (91*109*91, 3)

# Evaluate — result shape: (n_subjects, n_points, 1)
values_flat = fd_basis(grid_points_flat)
values_3d = values_flat[..., 0].reshape(-1, 121, 145, 121)

#############################
### Plot 5 slices ###
nrows = n_slices + 1
ncols = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
axes = axes.flatten()

for a in range(nrows):
    sns.heatmap(
        rand_image[indices[a], :, :],          # shape (109, 91)
        ax=axes[2*a],
        cmap="viridis",
        cbar=False,          # too cluttered with 91 colorbars
        xticklabels=False,
        yticklabels=False,
    )
    axes[2*a].set_title(f"Original, Slice={indices[a]+1}", fontsize=8)

    sns.heatmap(
        values_3d[0, indices[a], :, :],  # shape (109, 91)
        ax=axes[2*a+1],
        cmap="viridis",
        cbar=False,  # too cluttered with 91 colorbars
        xticklabels=False,
        yticklabels=False,
    )
    axes[2*a+1].set_title(f"Approximated, Slice={indices[a] + 1}", fontsize=8)

# Add a single shared colorbar
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(
    vmin=min(rand_image.min(), values_3d.min()), vmax=max(rand_image.max(),
                                                          values_3d.max())))
fig.colorbar(sm, ax=axes[:n_slices], shrink=0.5, label="Intensity")

plt.suptitle(f"Heatmaps of slices of the first axis for subject{random_subj}",
             fontsize=12,
             y=1.01)
plt.tight_layout()
plt.savefig("artifacts/heatmap_first_slice.png", dpi=150, bbox_inches="tight")
# plt.show()
