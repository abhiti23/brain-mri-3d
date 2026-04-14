import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.special import logsumexp

# --- Settings ---
input_file = 'group_average_gm_T1w.npy'
output_array_file = 'reconstructed_brain_500comp_weighted.npy' 
n_components = 500
threshold = 0.005

# Iteration fractions: Warm-up phase, followed by up to 100 full iterations.
# Early stopping will break the loop once it converges, so it rarely hits the full 100.
subsample_fractions = [0.6, 0.7, 0.8, 0.9] + [1.0] * 100 
# ----------------

def fit_weighted_gmm(X, W, n_components, fractions, reg_covar=1e-5, tol=1e-3):
    """
    Highly Optimized Weighted Gaussian Mixture Model using pure NumPy BLAS/LAPACK calls.
    Includes early stopping based on log-likelihood convergence.
    """
    total_N = X.shape[0]
    dim = X.shape[1]
    
    print("   -> Initializing parameters...")
    prob_dist = W / W.sum()
    init_idx = np.random.choice(total_N, n_components, replace=False, p=prob_dist)
    means = X[init_idx].astype(float)
    
    global_cov = np.cov(X.T)
    covars = np.tile(global_cov, (n_components, 1, 1))
    pi_weights = np.ones(n_components) / n_components
    
    reg_matrix = np.eye(dim) * reg_covar
    log_prob_const = -0.5 * dim * np.log(2 * np.pi)

    # Track previous log-likelihood for early stopping
    prev_ll = -np.inf

    for iteration, frac in enumerate(fractions):
        iter_start = time.time()
        
        if frac < 1.0:
            sample_size = int(frac * total_N)
            idx = np.random.choice(total_N, sample_size, replace=False)
            X_batch = X[idx]
            W_batch = W[idx]
        else:
            X_batch = X
            W_batch = W
            
        N_batch = X_batch.shape[0]
        
        # --- PRECOMPUTE PRECISION MATRICES IN C-LEVEL LAPACK ---
        regularized_covars = covars + reg_matrix
        precisions = np.linalg.inv(regularized_covars)
        _, log_dets = np.linalg.slogdet(regularized_covars)
        
        # --- BATCHED E-STEP ---
        log_resp = np.empty((N_batch, n_components))
        log_pi = np.log(pi_weights)
        
        for k in range(n_components):
            diff = X_batch - means[k]
            maha = np.sum(np.dot(diff, precisions[k]) * diff, axis=1)
            log_resp[:, k] = log_prob_const - 0.5 * (log_dets[k] + maha) + log_pi[k]
            
        log_prob_norm = logsumexp(log_resp, axis=1)
        
        # Calculate the average weighted log-likelihood for this iteration
        current_ll = np.average(log_prob_norm, weights=W_batch)
        
        log_resp -= log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp) 
        
        # --- BATCHED M-STEP ---
        weighted_resp = resp * W_batch[:, np.newaxis] 
        Nk = weighted_resp.sum(axis=0) + 1e-10 
        
        pi_weights = Nk / W_batch.sum()
        means = np.dot(weighted_resp.T, X_batch) / Nk[:, np.newaxis]
        
        for k in range(n_components):
            diff = X_batch - means[k]
            covars[k] = np.dot(diff.T, diff * weighted_resp[:, k, np.newaxis]) / Nk[k]
            
        # --- CONVERGENCE CHECK ---
        if frac == 1.0:
            ll_change = current_ll - prev_ll
            print(f"   -> Iteration {iteration+1} (Data: 100%) - Time: {time.time() - iter_start:.1f}s - LL Change: {ll_change:.6f}")
            
            # If the change in log-likelihood is smaller than the tolerance, stop early
            if abs(ll_change) < tol:
                print(f"\n   => SUCCESS: Converged at iteration {iteration+1} (Tolerance limit {tol} reached)!")
                break
                
            prev_ll = current_ll
        else:
            print(f"   -> Iteration {iteration+1} (Data: {frac*100:g}%) - Time: {time.time() - iter_start:.1f}s")

    return means, covars, pi_weights

def score_samples_gmm(X, means, covars, pi_weights, reg_covar=1e-5):
    """Evaluates final PDF using the same optimized BLAS routines."""
    N_batch = X.shape[0]
    n_components = len(means)
    dim = X.shape[1]
    
    reg_matrix = np.eye(dim) * reg_covar
    log_prob_const = -0.5 * dim * np.log(2 * np.pi)
    
    regularized_covars = covars + reg_matrix
    precisions = np.linalg.inv(regularized_covars)
    _, log_dets = np.linalg.slogdet(regularized_covars)
    
    log_resp = np.empty((N_batch, n_components))
    log_pi = np.log(pi_weights)
    
    for k in range(n_components):
        diff = X - means[k]
        maha = np.sum(np.dot(diff, precisions[k]) * diff, axis=1)
        log_resp[:, k] = log_prob_const - 0.5 * (log_dets[k] + maha) + log_pi[k]
        
    return logsumexp(log_resp, axis=1)

# ==========================================
# MAIN EXECUTION
# ==========================================

print("1. Loading, squeezing, and thresholding original data...")
brain_data = np.squeeze(np.load(input_file))
max_actual = np.max(brain_data[brain_data > threshold])

if os.path.exists(output_array_file):
    print(f"\n---> SUCCESS: Found existing reconstruction!")
    print(f"     Loading '{output_array_file}' directly...")
    reconstructed_brain = np.load(output_array_file)

else:
    print(f"\n---> No saved reconstruction found. Fitting Weighted GMM from scratch...")
    mask = brain_data > threshold
    coordinates = np.argwhere(mask)
    actual_intensities = brain_data[mask]

    print(f"2. Fitting {n_components} components using Optimized Weighted EM...")
    start_time = time.time()
    
    means, covars, pi_weights = fit_weighted_gmm(
        X=coordinates, 
        W=actual_intensities, 
        n_components=n_components, 
        fractions=subsample_fractions
    )
    
    print(f"   Total Fitting Done in {(time.time() - start_time)/60:.1f} minutes.")

    # Save the extracted centers to a separate file (Original setup preserved)
    centers_file = 'group_average_centers.npy'
    print(f"   Saving GMM centers to '{centers_file}'...")
    np.save(centers_file, means)

    # NEW: Save intensities (weights) and covariances to a separate file
    extra_params_file = 'group_average_covars_weights.npz'
    print(f"   Saving GMM covariances and intensities to '{extra_params_file}'...")
    np.savez(extra_params_file, covars=covars, weights=pi_weights)

    print("3. Reconstructing the 3D volume with Min-Max scaling...")
    log_pdf_values = score_samples_gmm(coordinates, means, covars, pi_weights)
    pdf_values = np.exp(log_pdf_values)
    
    predicted_intensities = (pdf_values / np.max(pdf_values)) * max_actual

    reconstructed_brain = np.zeros_like(brain_data)
    reconstructed_brain[tuple(coordinates.T)] = predicted_intensities

    print(f"   Saving reconstructed 3D array to '{output_array_file}'...")
    np.save(output_array_file, reconstructed_brain)
    print("   Save complete!\n")

# ----------------------------
print("4. Generating Sagittal, Coronal, and Axial plots...")
vmax = max_actual

def save_view_plot(axis, view_name, filename):
    dim_size = brain_data.shape[axis]
    slices_to_plot = [int(dim_size * 0.4), int(dim_size * 0.5), int(dim_size * 0.6)]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
    fig.suptitle(f"Original vs Weighted GMM ({n_components} Comp) - {view_name} View", fontsize=16)
    
    for i, slice_idx in enumerate(slices_to_plot):
        ax_orig = axes[i, 0]
        ax_recon = axes[i, 1]
        
        if axis == 0:   # Sagittal
            slice_orig = brain_data[slice_idx, :, :].T
            slice_recon = reconstructed_brain[slice_idx, :, :].T
        elif axis == 1: # Coronal
            slice_orig = brain_data[:, slice_idx, :].T
            slice_recon = reconstructed_brain[:, slice_idx, :].T
        else:           # Axial
            slice_orig = brain_data[:, :, slice_idx].T
            slice_recon = reconstructed_brain[:, :, slice_idx].T
            
        im_orig = ax_orig.imshow(slice_orig, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        ax_orig.set_title(f"Original Data (Slice {slice_idx})")
        ax_orig.axis('off')
        
        im_recon = ax_recon.imshow(slice_recon, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        ax_recon.set_title(f"Weighted GMM Recon (Slice {slice_idx})")
        ax_recon.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(im_orig, cax=cbar_ax, orientation='horizontal', label='Voxel Intensity')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(filename, dpi=300)
    plt.close() 
    print(f"   -> Saved {filename}")

save_view_plot(axis=0, view_name="Sagittal", filename="weighted_gmm_sagittal.png")
save_view_plot(axis=1, view_name="Coronal",  filename="weighted_gmm_coronal.png")
save_view_plot(axis=2, view_name="Axial",    filename="weighted_gmm_axial.png")

print("\nSuccess! Array saved and images generated.")
