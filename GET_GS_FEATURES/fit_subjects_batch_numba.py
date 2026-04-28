"""
Stage 2: per-subject weighted EM with means fixed to the average-brain
centers. Warm-started from the average-brain's covariances and mixing
weights, so EM converges in few iterations.

Performance tricks borrowed from the collaborator's Numba version:
  - Numba @njit on the E+M inner loops
  - Unrolled 3D Mahalanobis distance and outer products
  - Fused E-step + M-step accumulation in one pass over voxels
  - Warm-start initialization from average-brain fit

Deliberately NOT used:
  - float32 casting (kept as float64 for numerical reliability)
  - fastmath=True (kept IEEE 754 semantics to avoid silent NaN/Inf)

Output per subject: [pi_k * T, cxx, cxy, cxz, cyy, cyz, czz] * K  (7K features)
Multiplying pi by T embeds total GM volume into the per-component features,
so a linear regression on the 7K outputs has access to both relative and
absolute volume information (as we established earlier).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import time
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from numba import njit


# ==========================================
# Numba EM core (float64, no fastmath)
# ==========================================
@njit(cache=True)
def em_step_3d(X, W, fixed_means, precisions, log_dets, pi_weights, log_prob_const):
    """
    One EM iteration for weighted fixed-means GMM in 3D.
    Returns log_prob_norm (per voxel), new covariance estimates, and
    weighted component masses Nk (= sum_i w_i * gamma_ik).
    """
    N = X.shape[0]
    K = fixed_means.shape[0]

    resp = np.empty((N, K), dtype=np.float64)
    log_pi = np.log(pi_weights)
    log_prob_norm = np.empty(N, dtype=np.float64)

    # --- E-STEP ---
    for i in range(N):
        max_val = -1e300
        for k in range(K):
            diff0 = X[i, 0] - fixed_means[k, 0]
            diff1 = X[i, 1] - fixed_means[k, 1]
            diff2 = X[i, 2] - fixed_means[k, 2]

            # Mahalanobis: diff^T @ precision @ diff  (3x3, unrolled)
            maha = (
                diff0 * (diff0 * precisions[k, 0, 0]
                       + diff1 * precisions[k, 1, 0]
                       + diff2 * precisions[k, 2, 0])
              + diff1 * (diff0 * precisions[k, 0, 1]
                       + diff1 * precisions[k, 1, 1]
                       + diff2 * precisions[k, 2, 1])
              + diff2 * (diff0 * precisions[k, 0, 2]
                       + diff1 * precisions[k, 1, 2]
                       + diff2 * precisions[k, 2, 2])
            )

            val = log_prob_const - 0.5 * (log_dets[k] + maha) + log_pi[k]
            resp[i, k] = val
            if val > max_val:
                max_val = val

        # logsumexp across K for voxel i
        sum_exp = 0.0
        for k in range(K):
            sum_exp += np.exp(resp[i, k] - max_val)
        log_prob_norm[i] = max_val + np.log(sum_exp)

        # Convert to normalized responsibilities in-place
        for k in range(K):
            resp[i, k] = np.exp(resp[i, k] - log_prob_norm[i])

    # --- M-STEP ---
    new_covars = np.zeros((K, 3, 3), dtype=np.float64)
    Nk = np.zeros(K, dtype=np.float64)

    for i in range(N):
        w_i = W[i]
        diff_cache0 = 0.0
        diff_cache1 = 0.0
        diff_cache2 = 0.0
        for k in range(K):
            wr = resp[i, k] * w_i
            Nk[k] += wr

            diff0 = X[i, 0] - fixed_means[k, 0]
            diff1 = X[i, 1] - fixed_means[k, 1]
            diff2 = X[i, 2] - fixed_means[k, 2]

            # Weighted outer product (symmetric; we fill all 9 for simplicity)
            new_covars[k, 0, 0] += wr * diff0 * diff0
            new_covars[k, 0, 1] += wr * diff0 * diff1
            new_covars[k, 0, 2] += wr * diff0 * diff2
            new_covars[k, 1, 0] += wr * diff1 * diff0
            new_covars[k, 1, 1] += wr * diff1 * diff1
            new_covars[k, 1, 2] += wr * diff1 * diff2
            new_covars[k, 2, 0] += wr * diff2 * diff0
            new_covars[k, 2, 1] += wr * diff2 * diff1
            new_covars[k, 2, 2] += wr * diff2 * diff2

    for k in range(K):
        nk_safe = Nk[k] + 1e-12
        for a in range(3):
            for b in range(3):
                new_covars[k, a, b] /= nk_safe

    return log_prob_norm, new_covars, Nk


# ==========================================
# Python wrapper around the Numba core
# ==========================================
def fit_subject(coords, intensities, fixed_means, init_covars, init_pi,
                max_iter=100, tol=1e-4, reg_covar=1e-5):
    """
    Weighted EM with fixed means, warm-started from (init_covars, init_pi).
    Returns (pi, covars) after convergence.
    """
    dim = coords.shape[1]
    X = np.ascontiguousarray(coords, dtype=np.float64)
    W = np.ascontiguousarray(intensities, dtype=np.float64)
    means = np.ascontiguousarray(fixed_means, dtype=np.float64)
    covars = np.ascontiguousarray(init_covars, dtype=np.float64).copy()
    pi = np.ascontiguousarray(init_pi, dtype=np.float64).copy()

    reg_matrix = np.eye(dim, dtype=np.float64) * reg_covar
    log_prob_const = -0.5 * dim * np.log(2.0 * np.pi)
    W_sum = W.sum()
    prev_ll = -np.inf

    for it in range(max_iter):
        # Regularize, invert, slogdet — done in float64 numpy outside Numba
        reg_covars = covars + reg_matrix
        precisions = np.linalg.inv(reg_covars)
        _, log_dets = np.linalg.slogdet(reg_covars)

        log_prob_norm, covars, Nk = em_step_3d(
            X, W, means, precisions, log_dets, pi, log_prob_const
        )

        # Update mixing weights: intensity-weighted mass fraction per component
        pi = Nk / W_sum
        # Keep pi strictly positive for next iter's log
        pi = np.clip(pi, 1e-12, None)
        pi /= pi.sum()

        ll = np.sum(W * log_prob_norm) / W_sum
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return pi, covars, (it + 1)


# ==========================================
# Per-subject worker
# ==========================================
def process_subject(args):
    (file_path, fixed_means, init_covars, init_pi,
     threshold, output_dir, multiply_by_T) = args

    filename = os.path.basename(file_path)
    output_filepath = os.path.join(output_dir, filename)
    if os.path.exists(output_filepath):
        return f"SKIP: {filename}"

    try:
        brain = np.squeeze(np.load(file_path))
        mask = brain > threshold
        coords = np.argwhere(mask).astype(np.float64)
        intensities = brain[mask].astype(np.float64)
        T = intensities.sum()

        pi, covars, n_iter = fit_subject(
            coords, intensities, fixed_means, init_covars, init_pi
        )

        K = len(fixed_means)
        scale = T if multiply_by_T else 1.0
        params = np.empty(7 * K, dtype=np.float64)
        for k in range(K):
            c = covars[k]
            base = 7 * k
            params[base + 0] = pi[k] * scale
            params[base + 1] = c[0, 0]
            params[base + 2] = c[0, 1]
            params[base + 3] = c[0, 2]
            params[base + 4] = c[1, 1]
            params[base + 5] = c[1, 2]
            params[base + 6] = c[2, 2]

        np.save(output_filepath, params)
        return f"DONE: {filename}  iters={n_iter}  T={T:.1f}"

    except Exception as e:
        return f"ERROR: {filename} -> {e}"


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--pattern', default='../sub-*_preproc-cat12vbm_desc-gm_T1w.npy',
                        help="Glob for subject files (default looks one dir up)")
    parser.add_argument('--centers', default='centers.npy')
    parser.add_argument('--avg-model', default='avg_full_model.npz',
                        help="npz with keys: means, covariances, weights, total_intensity")
    parser.add_argument('--out-dir', default='gmm_features_weighted')
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--no-multiply-T', action='store_true',
                        help="Store raw pi_k instead of pi_k * T_subject")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load centers and warm-start parameters from average-brain fit
    fixed_means = np.load(args.centers).astype(np.float64)
    avg = np.load(args.avg_model)
    init_covars = avg['covariances'].astype(np.float64)
    init_pi = avg['weights'].astype(np.float64)

    # Sanity checks
    assert fixed_means.shape[0] == init_covars.shape[0] == init_pi.shape[0], \
        "centers, init_covars, init_pi must all have K components"
    assert np.allclose(fixed_means, avg['means']), \
        "centers.npy and avg_model['means'] disagree — are they from the same fit?"

    K = fixed_means.shape[0]
    print(f"[*] K = {K}, threshold = {args.threshold}")
    print(f"[*] Warm-start from {args.avg_model}")
    print(f"[*] Feature convention: pi * T = {not args.no_multiply_T}")

    all_files = sorted(glob.glob(args.pattern))
    if not all_files:
        print(f"[!] No files matching '{args.pattern}'")
        return
    files = all_files[args.start:(args.end if args.end else len(all_files))]
    n_total = len(files)

    n_workers = args.workers or cpu_count()
    print(f"[*] {n_total} subjects, {n_workers} workers")
    print(f"[*] First subject is slower (~2s JIT compile)\n")

    worker_args = [
        (fp, fixed_means, init_covars, init_pi,
         args.threshold, args.out_dir, not args.no_multiply_T)
        for fp in files
    ]

    t0 = time.time()
    completed = 0
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_subject, worker_args):
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed
            eta = (n_total - completed) / rate / 60 if rate > 0 else 0
            print(f"[{completed}/{n_total}] {result} | ETA: {eta:.1f} min")

    print(f"\n[*] Done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
