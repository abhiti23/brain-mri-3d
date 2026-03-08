# this file takes in 3d tensors and turns them into coefficients using a
# tensor product of B-splines. It iterates over all extracted VBM images,
# converts them into coefficients and stores them in the file
# "VBM_coefs.csv" in artifacts/
import numpy as np
import os
import skfda
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis, TensorBasis

def fd_coef(subj_number = 0, n_basis = 8, verbose = False):
    """
    Parameters
    -----------
    subj_number : int
        subject number for the brain image
    n_basis: int
        number of basis functions per dimension. The tensor product will
        have (n_basis)^3 number of basis functions

    Returns
    ---------------
    coefficients : array of size (n_basis)^3
        coefficients of fitted basis functions
    """
    file_name = f"data/raw/public_data_challenge/VBM_extracted/VBM_{subj_number}.npy"
    image = np.load(file_name)
    shape = image.shape

    # --- 2. Define the grid points along each axis (normalized to [0, 1]) ---
    grid_points = [np.linspace(0, 1, s) for s in shape]

    # --- 3. Wrap in FDataGrid ---
    # FDataGrid expects shape (n_samples, *grid_dims), so add a batch dim
    fd = FDataGrid(
        data_matrix=image[np.newaxis, ...],  # shape: (1, 64, 64, 64)
        grid_points=grid_points
    )

    # --- 4. Define tensor-product cubic B-spline basis ---
    n_basis_per_dim = n_basis  # tune this to control compression ratio
    bspline_univariate = BSplineBasis(domain_range=(0, 1),
                                             n_basis=n_basis_per_dim,
                                             order=4) # order 4 -> cubic spline
    basis = TensorBasis([bspline_univariate, bspline_univariate, bspline_univariate])

    # --- 5. Project onto basis (least-squares fit) ---
    fd_basis = fd.to_basis(basis)

    # --- 6. Extract coefficients ---
    coefficients = fd_basis.coefficients  # shape: (1, n_basis_per_dim**3)

    if verbose:
        print(f"Image number {subj_number}")
        print(f"Original voxels : {np.prod(shape)}")
        print(f"Coefficients    : {coefficients.shape[1]}")  # 512
        print(f"Compression ratio: {np.prod(shape) / coefficients.shape[1]:.0f}x")

    print(f"{subj_number} images processed")

    return coefficients

if __name__ == "__main__":

    VBM_DIR = "data/raw/public_data_challenge/VBM_extracted"
    #vbm_files = os.listdir(VBM_DIR)
    #L = len(vbm_files)
    L = 3227
    print(f"Found {L} VBM files.")

    n_basis = 8
    coef_matrix = np.zeros((L, n_basis**3), dtype = np.float64)

    # since code takes really long, let's add checkpoints
    checkpoint_path = "artifacts/checkpoint.npz"
    save_every = 100

    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        all_coefficients = list(checkpoint["coefficients"])
        start_idx = int(checkpoint["next_idx"])
        print(f"Resuming from file {start_idx}")
    else:
        all_coefficients = []
        start_idx = 0
        print("Starting fresh")

    for i in range(start_idx, L):
        all_coefficients.append(fd_coef(i))

        if (i + 1) % save_every == 0:
            np.savez(
                checkpoint_path,
                coefficients=np.array(all_coefficients),
                next_idx=i + 1
            )
            print(f"{i + 1} Checkpoint saved")

    final_coefficients = np.array(
        all_coefficients)  # shape: (n_files, n_basis**3)
    np.savez("artifacts/coefficients_final.npz",
             coefficients=final_coefficients)
    print(f"Done! Final shape: {final_coefficients.shape}")
