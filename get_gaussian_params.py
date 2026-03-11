import os
import glob
import time
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

# --- 1. CUSTOM GMM SUBCLASS ---
class FixedMeansGMM(GaussianMixture):
    def __init__(self, fixed_means, **kwargs):
        super().__init__(covariance_type='full', **kwargs)
        self.fixed_means = np.array(fixed_means)
        
    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self.means_ = self.fixed_means.copy()
        
    def _m_step(self, X, log_resp):
        """
        Custom M-step: Updates weights and covariances relative to the FIXED means.
        """
        n_samples = X.shape[0]
        resp = np.exp(log_resp)
        
        # 1. Update Weights (Intensities)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / n_samples
        
        # 2. Lock Means
        self.means_ = self.fixed_means.copy()
        
        # 3. Update Covariances relative to fixed means
        covariances = np.empty((self.n_components, X.shape[1], X.shape[1]))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[::X.shape[1] + 1] += self.reg_covar 
            
        self.covariances_ = covariances
        # Sklearn requires this precision matrix for the next E-step iteration
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')

# --- 2. MAIN PIPELINE ---
def main():
    parser = argparse.ArgumentParser(description="Batch process VBMs using a Fixed-Means GMM.")
    parser.add_argument('--start', type=int, default=0, help="Index of the first file to process (default: 0)")
    parser.add_argument('--end', type=int, default=None, help="Index to stop at (default: end of list)")
    args = parser.parse_args()

    # Settings
    output_dir = 'gmm_1d_outputs'
    average_brain_file = 'group_average_gm_T1w.npy'
    centers_file = os.path.join(output_dir, 'group_average_centers.npy')
    n_components = 500
    threshold = 0.005

    os.makedirs(output_dir, exist_ok=True)

    # --- STEP A: HANDLE THE CENTERS ---
    if os.path.exists(centers_file):
        print(f"[*] Found existing centers: {centers_file}")
        fixed_means = np.load(centers_file)
    else:
        print(f"[*] Centers not found. Fitting average brain ({average_brain_file}) to generate them...")
        avg_brain = np.squeeze(np.load(average_brain_file))
        mask = avg_brain > threshold
        coords = np.argwhere(mask)
        intensities = avg_brain[mask]

        start_time = time.time()
        gmm_avg = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm_avg.fit(coords, intensities)
        
        fixed_means = gmm_avg.means_
        np.save(centers_file, fixed_means)
        print(f"    -> Saved centers in {(time.time() - start_time)/60:.1f} mins.")

    # --- STEP B: PREPARE FILE LIST ---
    # Grab all subject files and sort them alphabetically so indexing is consistent
    search_pattern = 'sub-*_preproc-cat12vbm_desc-gm_T1w.npy'
    all_files = sorted(glob.glob(search_pattern))
    
    if not all_files:
        print(f"[!] No files matching '{search_pattern}' found in the current directory.")
        return

    # Slice the list based on user input
    end_idx = args.end if args.end is not None else len(all_files)
    files_to_process = all_files[args.start:end_idx]
    total_files = len(files_to_process)

    print(f"\n[*] Found {len(all_files)} total subjects. Processing index {args.start} to {end_idx} ({total_files} files).")
    print("-" * 50)

    # --- STEP C: PROCESS BATCH ---
    for i, file_path in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        output_filepath = os.path.join(output_dir, filename)

        # Skip if already processed
        if os.path.exists(output_filepath):
            print(f"[{i+1}/{total_files}] Skipping {filename} (Already exists)")
            continue

        print(f"[{i+1}/{total_files}] Processing: {filename} ...")
        loop_start = time.time()

        # Load and prep data
        brain_data = np.squeeze(np.load(file_path))
        mask = brain_data > threshold
        coords = np.argwhere(mask)
        intensities = brain_data[mask]

        # Fit custom model
        fixed_model = FixedMeansGMM(fixed_means=fixed_means, n_components=n_components, random_state=42)
        fixed_model.fit(coords, intensities)

        # Extract parameters into structured 1D array
        params_1d = []
        for k in range(n_components):
            weight = fixed_model.weights_[k]
            cov = fixed_model.covariances_[k]
            
            # Extract upper triangle of symmetric 3x3 matrix
            c_xx, c_xy, c_xz = cov[0, 0], cov[0, 1], cov[0, 2]
            c_yy, c_yz = cov[1, 1], cov[1, 2]
            c_zz = cov[2, 2]
            
            # Append block of 7
            params_1d.extend([weight, c_xx, c_xy, c_xz, c_yy, c_yz, c_zz])

        params_1d = np.array(params_1d)
        np.save(output_filepath, params_1d)

        # Timers and ETA
        elapsed = time.time() - loop_start
        files_remaining = total_files - (i + 1)
        eta_mins = (files_remaining * elapsed) / 60
        
        print(f"    -> Saved. Took {elapsed:.1f}s. (Est. remaining: {eta_mins:.1f} mins)")

    print("\n[*] Batch complete!")

if __name__ == "__main__":
    main()