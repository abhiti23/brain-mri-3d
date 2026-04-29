"""
fd_coef_worker.py
-----------------
Processes a single VBM subject and saves its B-spline coefficients.
Uses a sorted file list so SLURM array index maps to a real filename.

Called by SLURM array jobs via:
    python fd_coef_worker.py <subj_index> --data_dir <path>
"""

import numpy as np
import sys
import os
import argparse
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis, TensorBasis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subj_index", type=int,
                        help="Index into the sorted list of VBM .npy files")
    parser.add_argument("--n_basis", type=int, default=8)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing VBM .npy files")
    parser.add_argument("--out_dir", type=str, default="artifacts/coefs",
                        help="Output directory for per-subject coefficient .npy files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build sorted list of all .npy files in the VBM directory
    all_files = sorted([
        f for f in os.listdir(args.data_dir)
        if f.endswith(".npy")
    ])

    if args.subj_index >= len(all_files):
        print(f"Error: index {args.subj_index} out of range "
              f"(found {len(all_files)} files)")
        sys.exit(1)

    filename = all_files[args.subj_index]
    # e.g. sub-100536464191_preproc-cat12vbm_desc-gm_T1w
    subj_id = filename.replace(".npy", "")

    # Output: coef_0001_sub-100536464191_preproc-cat12vbm_desc-gm_T1w.npy
    out_path = os.path.join(args.out_dir,
                            f"coef_{args.subj_index:04d}_{subj_id}.npy")

    # Skip if already done — safe to re-run failed jobs
    if os.path.exists(out_path):
        print(f"[{args.subj_index}] Already exists, skipping.")
        sys.exit(0)

    filepath = os.path.join(args.data_dir, filename)
    print(f"[{args.subj_index}] Loading {filename}")

    image = np.load(filepath)
    image = image.squeeze()  # (1, 1, 121, 145, 121) -> (121, 145, 121)
    shape = image.shape

    # Normalize grid to [0, 1] per axis
    grid_points = [np.linspace(0, 1, s) for s in shape]

    # Wrap as functional data object
    fd = FDataGrid(
        data_matrix=image[np.newaxis, ...],
        grid_points=grid_points
    )

    # Tensor-product cubic B-spline basis (order=4 -> cubic)
    bspline = BSplineBasis(domain_range=(0, 1), n_basis=args.n_basis, order=4)
    basis = TensorBasis([bspline, bspline, bspline])

    # Least-squares projection
    fd_basis = fd.to_basis(basis)
    coef = fd_basis.coefficients  # shape: (1, n_basis**3)

    if args.verbose:
        print(f"  File       : {filename}")
        print(f"  Voxels     : {np.prod(shape)}")
        print(f"  Coefs      : {coef.shape[1]}")
        print(f"  Compression: {np.prod(shape) / coef.shape[1]:.0f}x")

    np.save(out_path, coef)
    print(f"[{args.subj_index}] Saved -> {out_path}")
