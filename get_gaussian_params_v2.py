import os
import glob
import time
import argparse
import numpy as np
from scipy.special import logsumexp
from multiprocessing import Pool, cpu_count
from numba import njit

# Prevent numpy/OpenBLAS from over-subscribing threads 
# since we are parallelizing at the subject level with multiprocessing.Pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==========================================
# 1. NUMBA JIT COMPILED CORE ENGINE
# ==========================================
# This function compiles down to pure C/C++ machine code.
# The first time it runs, it will take ~2 seconds to compile. 
# After that, it executes at hardware maximum speed.

@njit(fastmath=True)
def numba_em_core_3d(X, W, fixed_means, precisions, log_dets, pi_weights, log_prob_const):
    N = X.shape[0]
    K = fixed_means.shape[0]
    
    resp = np.empty((N, K), dtype=np.float32)
    log_pi = np.log(pi_weights)
    log_prob_norm = np.empty(N, dtype=np.float32)
    
    # --- E-STEP ---
    for i in range(N):
        max_val = -1e30 
        for k in range(K):
            # Unrolled 3D distance
            diff0 = X[i, 0] - fixed_means[k, 0]
            diff1 = X[i, 1] - fixed_means[k, 1]
            diff2 = X[i, 2] - fixed_means[k, 2]
            
            # Unrolled Mahalanobis Matrix Multiplication (blazing fast)
            maha = (diff0 * (diff0 * precisions[k, 0, 0] + diff1 * precisions[k, 1, 0] + diff2 * precisions[k, 2, 0]) +
                    diff1 * (diff0 * precisions[k, 0, 1] + diff1 * precisions[k, 1, 1] + diff2 * precisions[k, 2, 1]) +
                    diff2 * (diff0 * precisions[k, 0, 2] + diff1 * precisions[k, 1, 2] + diff2 * precisions[k, 2, 2]))
            
            val = log_prob_const - 0.5 * (log_dets[k] + maha) + log_pi[k]
            resp[i, k] = val
            if val > max_val:
                max_val = val
        
        # logsumexp
        sum_exp = 0.0
        for k in range(K):
            sum_exp += np.exp(resp[i, k] - max_val)
        
        log_prob_norm[i] = max_val + np.log(sum_exp)
        
        # Convert log_resp to standard probabilities in place
        for k in range(K):
            resp[i, k] = np.exp(resp[i, k] - log_prob_norm[i])

    # --- M-STEP ---
    new_covars = np.zeros((K, 3, 3), dtype=np.float32)
    Nk = np.zeros(K, dtype=np.float32)
    
    for i in range(N):
        w_i = W[i]
        for k in range(K):
            w_resp = resp[i, k] * w_i
            Nk[k] += w_resp
            
            diff0 = X[i, 0] - fixed_means[k, 0]
            diff1 = X[i, 1] - fixed_means[k, 1]
            diff2 = X[i, 2] - fixed_means[k, 2]
            
            # Unrolled outer product summation
            new_covars[k, 0, 0] += w_resp * diff0 * diff0
            new_covars[k, 0, 1] += w_resp * diff0 * diff1
            new_covars[k, 0, 2] += w_resp * diff0 * diff2
            new_covars[k, 1, 0] += w_resp * diff1 * diff0
            new_covars[k, 1, 1] += w_resp * diff1 * diff1
            new_covars[k, 1, 2] += w_resp * diff1 * diff2
            new_covars[k, 2, 0] += w_resp * diff2 * diff0
            new_covars[k, 2, 1] += w_resp * diff2 * diff1
            new_covars[k, 2, 2] += w_resp * diff2 * diff2

    # Normalize covariances
    for k in range(K):
        nk_k = Nk[k] + 1e-10
        for d1 in range(3):
            for d2 in range(3):
                new_covars[k, d1, d2] /= nk_k
                
    return log_prob_norm, new_covars, Nk

# ==========================================
# 2. PYTHON WRAPPERS
# ==========================================

def fit_subject_fixed_means_numba(X, W, fixed_means, init_covars, init_weights, max_iter=100, tol=1e-3, reg_covar=1e-5):
    """Wraps the Numba engine with data casting and convergence checking."""
    dim = X.shape[1]
    n_components = len(fixed_means)
    
    # 1. Strict cast to float32 for maximum memory throughput
    X = X.astype(np.float32)
    W = W.astype(np.float32)
    fixed_means = fixed_means.astype(np.float32)
    pi_weights = init_weights.astype(np.float32)
    covars = init_covars.astype(np.float32)
    
    reg_matrix = (np.eye(dim) * reg_covar).astype(np.float32)
    log_prob_const = np.float32(-0.5 * dim * np.log(2 * np.pi))
    prev_ll = -np.inf

    for iteration in range(max_iter):
        regularized_covars = covars + reg_matrix
        precisions = np.linalg.inv(regularized_covars).astype(np.float32)
        _, log_dets = np.linalg.slogdet(regularized_covars)
        log_dets = log_dets.astype(np.float32)
        
        # Call compiled C++ equivalent
        log_prob_norm, covars, Nk = numba_em_core_3d(
            X, W, fixed_means, precisions, log_dets, pi_weights, log_prob_const
        )
        
        # Convergence Check
        current_ll = np.average(log_prob_norm, weights=W)
        pi_weights = Nk / W.sum()
            
        if abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return pi_weights, covars


# ==========================================
# 3. SINGLE SUBJECT WORKER
# ==========================================
def process_subject(args):
    file_path, fixed_means, init_covars, init_weights, threshold, output_dir = args
    filename = os.path.basename(file_path)
    output_filepath = os.path.join(output_dir, filename)

    if os.path.exists(output_filepath):
        return f"SKIP: {filename}"

    try:
        brain_data = np.squeeze(np.load(file_path))
        mask = brain_data > threshold
        coords = np.argwhere(mask)
        intensities = brain_data[mask]

        # Use the Numba backend
        final_weights, final_covars = fit_subject_fixed_means_numba(
            X=coords, 
            W=intensities, 
            fixed_means=fixed_means, 
            init_covars=init_covars, 
            init_weights=init_weights
        )

        params_1d = []
        n_components = len(fixed_means)
        for k in range(n_components):
            weight = final_weights[k]
            cov = final_covars[k]
            c_xx, c_xy, c_xz = cov[0, 0], cov[0, 1], cov[0, 2]
            c_yy, c_yz = cov[1, 1], cov[1, 2]
            c_zz = cov[2, 2]
            params_1d.extend([weight, c_xx, c_xy, c_xz, c_yy, c_yz, c_zz])

        params_1d = np.array(params_1d)
        np.save(output_filepath, params_1d)
        return f"DONE: {filename}"

    except Exception as e:
        return f"ERROR: {filename} -> {e}"


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Batch process VBMs using a Numba JIT Compiled Fixed-Means GMM.")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of parallel workers (default: all available CPUs)")
    args = parser.parse_args()

    output_dir = 'gmm_1d_outputs'
    threshold = 0.005

    os.makedirs(output_dir, exist_ok=True)

    # --- STEP A: HANDLE PRIORS ---
    centers_file = 'group_average_centers.npy'
    extra_params_file = 'group_average_covars_weights.npz'

    if os.path.exists(centers_file) and os.path.exists(extra_params_file):
        print(f"[*] Loading existing average priors from '{centers_file}' and '{extra_params_file}'...")
        fixed_means = np.load(centers_file)
        
        extra_data = np.load(extra_params_file)
        init_covars = extra_data['covars']
        init_weights = extra_data['weights']
    else:
        raise FileNotFoundError(
            f"[!] Missing prior files. Ensure '{centers_file}' and '{extra_params_file}' exist in the directory."
        )

    # --- STEP B: PREPARE FILES ---
    search_pattern = 'sub-*_preproc-cat12vbm_desc-gm_T1w.npy'
    all_files = sorted(glob.glob(search_pattern))
    
    if not all_files:
        print(f"[!] No files matching '{search_pattern}' found.")
        return

    end_idx = args.end if args.end is not None else len(all_files)
    files_to_process = all_files[args.start:end_idx]
    total_files = len(files_to_process)

    n_workers = args.workers if args.workers else cpu_count()
    print(f"\n[*] Processing {total_files} subjects using {n_workers} parallel workers.")
    print(f"[*] Note: The first subject may take an extra ~2s to JIT compile the C++ core.")
    print("-" * 65)

    # --- STEP C: PARALLEL EXECUTION ---
    worker_args = [
        (fp, fixed_means, init_covars, init_weights, threshold, output_dir)
        for fp in files_to_process
    ]

    start_time = time.time()
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_subject, worker_args):
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed  
            remaining = total_files - completed
            eta_mins = (remaining / rate) / 60 if rate > 0 else 0
            print(f"[{completed}/{total_files}] {result} | ETA: {eta_mins:.1f} mins")

    print(f"\n[*] Batch complete! Total time: {(time.time() - start_time)/60:.1f} mins")

if __name__ == "__main__":
    main()
