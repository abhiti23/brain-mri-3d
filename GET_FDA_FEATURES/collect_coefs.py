"""
collect_coefs.py
----------------
Run AFTER all SLURM array jobs finish.
Stacks all coef_NNNN_<subj_id>.npy files into a single matrix.
Also saves a subject ID list so rows can be matched back to subjects.

Usage:
    python collect_coefs.py --coef_dir <path> --out <path>
"""

import numpy as np
import os
import argparse
import glob
import sys


def collect(coef_dir, out_path):
    # Match files like coef_0001_sub-XXXXX_....npy
    pattern = os.path.join(coef_dir, "coef_*.npy")
    files = sorted(glob.glob(pattern), key=lambda f: int(
        os.path.basename(f).split("_")[1]  # sort by 4-digit index
    ))

    if len(files) == 0:
        print(f"No coefficient files found in '{coef_dir}'. Did the jobs finish?")
        sys.exit(1)

    print(f"Found {len(files)} coefficient files.")

    # Check for missing indices
    indices = [int(os.path.basename(f).split("_")[1]) for f in files]
    expected = set(range(min(indices), max(indices) + 1))
    missing = expected - set(indices)
    if missing:
        print(f"WARNING: {len(missing)} missing subjects: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        ans = input("Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit(1)

    # Stack into (n_subjects, n_coefs)
    arrays = []
    subj_ids = []
    for f in files:
        coef = np.load(f, allow_pickle=True)           # shape: (1, n_basis**3)
        arrays.append(coef)
        # Extract subject ID from filename: coef_0001_sub-XXXXX_....npy
        basename = os.path.basename(f).replace(".npy", "")
        # Everything after the 5-char "coef_NNNN_" prefix
        subj_id = "_".join(basename.split("_")[2:])
        subj_ids.append(subj_id)

    final = np.vstack(arrays)       # shape: (n_subjects, n_basis**3)
    print(f"Final coefficient matrix shape: {final.shape}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save .npz with both coefficients and subject IDs
    np.savez(out_path, coefficients=final, subject_ids=np.array(subj_ids))
    print(f"Saved -> {out_path}")

    # Save CSV of coefficients
    csv_path = out_path.replace(".npz", ".csv")
    np.savetxt(csv_path, final, delimiter=",")
    print(f"Saved -> {csv_path}")

    # Save subject ID list for reference
    id_path = out_path.replace(".npz", "_subject_ids.txt")
    with open(id_path, "w") as f:
        f.write("\n".join(subj_ids))
    print(f"Saved -> {id_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef_dir", type=str, required=True,
                        help="Directory containing coef_*.npy files")
    parser.add_argument("--out", type=str, required=True,
                        help="Output path for final .npz (e.g. artifacts/coefficients_final.npz)")
    args = parser.parse_args()

    collect(coef_dir=args.coef_dir, out_path=args.out)
