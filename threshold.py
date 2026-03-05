import numpy as np
import time
from sklearn.mixture import GaussianMixture

# --- Settings ---
input_file = 'group_average_gm_T1w.npy'
n_components = 100
thresholds_to_test = [0.0001, 0.0005, 0.001, 0.005, 0.01]
# ----------------

print("Loading brain data...")
brain_data = np.load(input_file)

# Dictionary to store our results for the final table
results = []

print(f"Beginning evaluation of {len(thresholds_to_test)} thresholds...\n")

for thresh in thresholds_to_test:
    print(f"--- Testing Threshold: {thresh} ---")
    start_time = time.time()
    
    # 1. Isolate the data for this specific threshold
    mask = brain_data > thresh
    coordinates = np.argwhere(mask)
    actual_intensities = brain_data[mask]
    
    num_voxels = coordinates.shape[0]
    if num_voxels == 0:
        print("  No voxels found above this threshold. Skipping...")
        continue
        
    total_intensity_mass = np.sum(actual_intensities)
    mean_actual_intensity = np.mean(actual_intensities)
    
    print(f"  Valid Voxels: {num_voxels}")
    print("  Fitting GMM...")
    
    # 2. Fit the model
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(coordinates, actual_intensities)
    
    # 3. Reconstruct the intensities
    print("  Reconstructing and calculating metrics...")
    pdf_values = np.exp(gmm.score_samples(coordinates))
    predicted_intensities = pdf_values * total_intensity_mass
    
    # 4. Calculate Metrics
    mse = np.mean((actual_intensities - predicted_intensities) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / mean_actual_intensity
    
    correlation_matrix = np.corrcoef(actual_intensities, predicted_intensities)
    pearson_r = correlation_matrix[0, 1]
    
    elapsed_time = time.time() - start_time
    print(f"  Done in {elapsed_time:.1f} seconds.\n")
    
    # 5. Save results for the table
    results.append({
        'Threshold': thresh,
        'Voxels': num_voxels,
        'Pearson_r': pearson_r,
        'NRMSE': nrmse,
        'RMSE': rmse
    })

# --- FINAL SUMMARY TABLE ---
print("="*75)
print(f"{'Threshold':<12} | {'Voxels':<10} | {'Pearson r':<12} | {'NRMSE':<12} | {'RMSE'}")
print("-" * 75)

for res in results:
    thresh_str = f"{res['Threshold']}"
    vox_str = f"{res['Voxels']}"
    r_str = f"{res['Pearson_r']:.4f}"
    nrmse_str = f"{res['NRMSE']:.4f}"
    rmse_str = f"{res['RMSE']:.2f}"
    
    print(f"{thresh_str:<12} | {vox_str:<10} | {r_str:<12} | {nrmse_str:<12} | {rmse_str}")
print("="*75)