"""
Forward pass: reconstruct each subject's 3D brain from their fitted GMM
parameters and compare against the original MRI.

For each subject, loads the 7K parameter vector [pi*T, cxx, cxy, cxz, cyy, cyz, czz]
per component, unpacks into weights and covariances, evaluates the mixture
on the full voxel grid, and reports similarity metrics.

Outputs:
  - Per-subject reconstructed .npy (optional, controlled by --save-recon)
  - CSV summary of all similarity metrics across subjects
  - Printed table to stdout
"""

import os
import glob
import time
import argparse
import numpy as np
from scipy.stats import multivariate_normal


def unpack_params(params_1d, n_components):
    """Unpack the 7K vector into (scaled_weights, covariances)."""
    weights = np.empty(n_components, dtype=np.float64)
    covars = np.empty((n_components, 3, 3), dtype=np.float64)

    for k in range(n_components):
        base = 7 * k
        weights[k] = params_1d[base + 0]  # pi_k * T
        c_xx = params_1d[base + 1]
        c_xy = params_1d[base + 2]
        c_xz = params_1d[base + 3]
        c_yy = params_1d[base + 4]
        c_yz = params_1d[base + 5]
        c_zz = params_1d[base + 6]
        covars[k] = [[c_xx, c_xy, c_xz],
                      [c_xy, c_yy, c_yz],
                      [c_xz, c_yz, c_zz]]

    # Recover T and pi from scaled weights
    T = weights.sum()
    pi = weights / T

    return pi, covars, T


def reconstruct_brain(pi, covars, means, T, brain_shape, chunk_size=200_000):
    """Evaluate f_hat(x) = T * sum_k pi_k * N(x | mu_k, Sigma_k) on full grid."""
    nx, ny, nz = brain_shape
    ii, jj, kk = np.mgrid[0:nx, 0:ny, 0:nz]
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float64)

    K = len(pi)
    recon = np.zeros(coords.shape[0], dtype=np.float64)

    # Precompute frozen distributions
    dists = []
    for k in range(K):
        try:
            dists.append(multivariate_normal(
                mean=means[k], cov=covars[k], allow_singular=True))
        except Exception as e:
            print(f"      [!] Component {k} failed: {e}")
            dists.append(None)

    n_chunks = (coords.shape[0] + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, coords.shape[0], chunk_size)):
        end = min(start + chunk_size, coords.shape[0])
        chunk = coords[start:end]
        acc = np.zeros(end - start, dtype=np.float64)
        for k in range(K):
            if dists[k] is None:
                continue
            acc += pi[k] * dists[k].pdf(chunk)
        recon[start:end] = acc
        if (ci + 1) % 3 == 0 or ci == n_chunks - 1:
            print(f"      chunk {ci + 1}/{n_chunks}")

    recon *= T
    return recon.reshape(brain_shape)


def similarity_metrics(original, reconstruction, mask):
    """Compute metrics on suprathreshold voxels."""
    a = original[mask].astype(np.float64)
    b = reconstruction[mask].astype(np.float64)

    pearson = np.corrcoef(a, b)[0, 1]
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ev = 1.0 - np.var(a - b) / np.var(a)
    l1 = np.sum(np.abs(a - b)) / np.sum(a)
    rmse = np.sqrt(np.mean((a - b) ** 2))
    mass_ratio = b.sum() / a.sum()

    return {
        "pearson_r": pearson,
        "cosine": cosine,
        "explained_variance": ev,
        "normalized_L1": l1,
        "rmse": rmse,
        "mass_ratio": mass_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', default='gmm_features_weighted',
                        help="Dir with per-subject 7K .npy feature files")
    parser.add_argument('--mri-pattern', default='../sub-*_preproc-cat12vbm_desc-gm_T1w.npy',
                        help="Glob for original MRI files")
    parser.add_argument('--centers', default='centers.npy')
    parser.add_argument('--n-components', type=int, default=500)
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--save-recon', action='store_true',
                        help="Save reconstructed .npy files (large, slow I/O)")
    parser.add_argument('--recon-dir', default='reconstructions')
    parser.add_argument('--csv-out', default='reconstruction_metrics.csv')
    args = parser.parse_args()

    means = np.load(args.centers).astype(np.float64)
    assert means.shape == (args.n_components, 3)

    # Match feature files to original MRIs by filename
    feature_files = sorted(glob.glob(os.path.join(args.features_dir, '*.npy')))
    mri_files = sorted(glob.glob(args.mri_pattern))
    mri_lookup = {os.path.basename(f): f for f in mri_files}

    pairs = []
    for ff in feature_files:
        fname = os.path.basename(ff)
        if fname in mri_lookup:
            pairs.append((ff, mri_lookup[fname]))
        else:
            print(f"[!] No matching MRI for {fname}, skipping")

    print(f"[*] {len(pairs)} matched subject pairs")
    print(f"[*] Centers: {args.centers} ({args.n_components} components)")
    if args.save_recon:
        os.makedirs(args.recon_dir, exist_ok=True)
        print(f"[*] Saving reconstructions to {args.recon_dir}/")

    # Header
    metric_names = ["pearson_r", "cosine", "explained_variance",
                    "normalized_L1", "rmse", "mass_ratio"]
    header = "subject," + ",".join(metric_names)

    results = []
    t0 = time.time()

    for i, (feat_path, mri_path) in enumerate(pairs):
        fname = os.path.basename(feat_path)
        sub_id = fname.split('_')[0]
        print(f"\n[{i+1}/{len(pairs)}] {sub_id}")

        # Load and unpack
        params = np.load(feat_path)
        pi, covars, T = unpack_params(params, args.n_components)
        print(f"   T = {T:.1f}, pi range = [{pi.min():.6f}, {pi.max():.6f}]")

        # Load original
        original = np.squeeze(np.load(mri_path))
        mask = original > args.threshold

        # Reconstruct
        print(f"   Reconstructing ({original.shape})...")
        t_sub = time.time()
        recon = reconstruct_brain(pi, covars, means, T, original.shape)
        print(f"   Done in {(time.time() - t_sub)/60:.1f} min")

        # Save reconstruction if requested
        if args.save_recon:
            recon_path = os.path.join(args.recon_dir, fname)
            np.save(recon_path, recon)
            print(f"   Saved {recon_path}")

        # Metrics
        m = similarity_metrics(original, recon, mask)
        row = [sub_id] + [f"{m[k]:.6f}" for k in metric_names]
        results.append(row)

        print(f"   pearson={m['pearson_r']:.4f}  cosine={m['cosine']:.4f}  "
              f"EV={m['explained_variance']:.4f}  L1={m['normalized_L1']:.4f}  "
              f"mass={m['mass_ratio']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'SUBJECT':>20}  {'pearson':>8}  {'cosine':>8}  {'EV':>8}  "
          f"{'norm_L1':>8}  {'RMSE':>8}  {'mass':>8}")
    print(f"{'-'*70}")
    for row in results:
        print(f"{row[0]:>20}  " + "  ".join(f"{v:>8}" for v in row[1:]))

    # Averages
    numeric = np.array([[float(v) for v in row[1:]] for row in results])
    means_row = numeric.mean(axis=0)
    stds_row = numeric.std(axis=0)
    print(f"{'-'*70}")
    print(f"{'MEAN':>20}  " + "  ".join(f"{v:>8.4f}" for v in means_row))
    print(f"{'STD':>20}  " + "  ".join(f"{v:>8.4f}" for v in stds_row))

    # Save CSV
    with open(args.csv_out, 'w') as f:
        f.write(header + "\n")
        for row in results:
            f.write(",".join(row) + "\n")
    print(f"\n[*] Metrics saved to {args.csv_out}")
    print(f"[*] Total time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
