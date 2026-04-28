"""
Reconstruct a 3D Gaussian brain from a saved fitted model (NO re-fitting).

Loads means, covariances, weights, and total intensity from the .npz
produced by fit_average_brain_free_means.py --save-full, then evaluates

    f_hat(x) = T * sum_k pi_k * N(x | mu_k, Sigma_k)

on the full voxel grid. Compares against the original brain using several
similarity metrics.

This is pure forward evaluation — should be fast (a few minutes, dominated
by 500 Gaussian pdf evaluations at ~1.8M voxels).
"""

import argparse
import numpy as np


def reconstruct_brain(means, covariances, weights, total_intensity,
                      brain_shape, chunk_size=200_000):
    from scipy.stats import multivariate_normal

    nx, ny, nz = brain_shape
    ii, jj, kk = np.mgrid[0:nx, 0:ny, 0:nz]
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float64)

    recon = np.zeros(coords.shape[0], dtype=np.float64)
    K = len(weights)

    # Freeze pdfs once — reused across chunks.
    dists = []
    for k in range(K):
        try:
            dists.append(multivariate_normal(
                mean=means[k], cov=covariances[k], allow_singular=True))
        except Exception as e:
            print(f"[!] Component {k} failed: {e}")
            dists.append(None)

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        chunk = coords[start:end]
        acc = np.zeros(end - start, dtype=np.float64)
        for k in range(K):
            if dists[k] is None:
                continue
            acc += weights[k] * dists[k].pdf(chunk)
        recon[start:end] = acc
        if (start // chunk_size) % 5 == 0:
            print(f"    voxels {end}/{coords.shape[0]}")

    recon *= total_intensity
    return recon.reshape(brain_shape)


def similarity_metrics(original, reconstruction, mask=None):
    if mask is not None:
        a = original[mask].astype(np.float64)
        b = reconstruction[mask].astype(np.float64)
    else:
        a = original.ravel().astype(np.float64)
        b = reconstruction.ravel().astype(np.float64)

    return {
        "pearson_r":           np.corrcoef(a, b)[0, 1],
        "cosine":              np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        "explained_variance":  1.0 - np.var(a - b) / np.var(a),
        "normalized_L1_error": np.sum(np.abs(a - b)) / np.sum(a),
        "rmse":                np.sqrt(np.mean((a - b) ** 2)),
        "total_original":      a.sum(),
        "total_reconstructed": b.sum(),
        "mass_ratio":          b.sum() / a.sum(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="Path to .npz saved by --save-full")
    parser.add_argument('brain', help="Path to original 3D brain .npy for comparison")
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--save-recon', default=None,
                        help="Path to save reconstructed 3D .npy (for plotting)")
    args = parser.parse_args()

    print(f"[*] Loading saved model from {args.model}")
    m = np.load(args.model)
    means = m['means']
    covariances = m['covariances']
    weights = m['weights']
    total_intensity = float(m['total_intensity'])
    print(f"    K={len(weights)}, T={total_intensity:.2f}")

    print(f"[*] Loading original brain {args.brain}")
    original = np.squeeze(np.load(args.brain))
    print(f"    shape={original.shape}")

    print(f"\n[*] Reconstructing on full voxel grid...")
    recon = reconstruct_brain(means, covariances, weights,
                              total_intensity, original.shape)

    if args.save_recon:
        np.save(args.save_recon, recon)
        print(f"[*] Saved reconstruction to {args.save_recon}")

    mask = original > args.threshold

    print("\n=== Similarity on suprathreshold mask ===")
    for k, v in similarity_metrics(original, recon, mask=mask).items():
        print(f"  {k:24s} {v:.6f}")

    print("\n=== Similarity on whole volume ===")
    for k, v in similarity_metrics(original, recon, mask=None).items():
        print(f"  {k:24s} {v:.6f}")


if __name__ == "__main__":
    main()
