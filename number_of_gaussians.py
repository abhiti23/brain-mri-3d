import numpy as np
import time
from sklearn.mixture import GaussianMixture

# --- Settings ---
input_file = 'group_average_gm_T1w.npy'
threshold = 0.005  # Fixed to your optimized sweet spot
components_to_test = [150, 200, 300, 350, 400, 450] # Adjust these as your CPU allows
# ----------------

print("1. Loading and thresholding data...")
brain_data = np.load(input_file)
mask = brain_data > threshold
coordinates = np.argwhere(mask)
actual_intensities = brain_data[mask]

num_voxels = coordinates.shape[0]
total_intensity_mass = np.sum(actual_intensities)
mean_actual_intensity = np.mean(actual_intensities)

print(f"   Fixed Threshold : {threshold}")
print(f"   Valid Voxels    : {num_voxels}")
print(f"   Total Mass      : {total_intensity_mass:.2f}\n")

results = []

print(f"Beginning evaluation of {len(components_to_test)} different GMM sizes...")

for n_comp in components_to_test:
    print(f"\n--- Fitting {n_comp} Gaussians ---")
    start_time = time.time()
    
    # Fit the model
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
    gmm.fit(coordinates, actual_intensities)
    
    # Reconstruct the intensities
    pdf_values = np.exp(gmm.score_samples(coordinates))
    predicted_intensities = pdf_values * total_intensity_mass
    
    # Calculate Metrics
    mse = np.mean((actual_intensities - predicted_intensities) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / mean_actual_intensity
    
    correlation_matrix = np.corrcoef(actual_intensities, predicted_intensities)
    pearson_r = correlation_matrix[0, 1]
    
    # Track time
    elapsed_time = time.time() - start_time
    print(f"  Done! Time taken: {elapsed_time:.1f} seconds. (Pearson r = {pearson_r:.4f})")
    
    # Save results for the final table
    results.append({
        'Components': n_comp,
        'Time_sec': elapsed_time,
        'Pearson_r': pearson_r,
        'NRMSE': nrmse
    })

# --- FINAL SUMMARY TABLE ---
print("\n" + "="*65)
print(f"{'Components':<12} | {'Time (sec)':<12} | {'Pearson r':<12} | {'NRMSE'}")
print("-" * 65)

for res in results:
    comp_str = f"{res['Components']}"
    time_str = f"{res['Time_sec']:.1f}"
    r_str = f"{res['Pearson_r']:.4f}"
    nrmse_str = f"{res['NRMSE']:.4f}"
    
    print(f"{comp_str:<12} | {time_str:<12} | {r_str:<12} | {nrmse_str}")
print("="*65)