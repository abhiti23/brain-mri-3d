import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
import os

# --- Settings ---
input_file = 'group_average_gm_T1w.npy'
output_array_file = 'reconstructed_brain_500comp.npy' # The file we will save/load
n_components = 500
threshold = 0.005
# ----------------

print("1. Loading, squeezing, and thresholding original data...")
brain_data = np.squeeze(np.load(input_file))

# We need the max intensity for the colorbar, regardless of whether we fit or load
max_actual = np.max(brain_data[brain_data > threshold])

# --- THE TIME-SAVER BLOCK ---
if os.path.exists(output_array_file):
    print(f"\n---> SUCCESS: Found existing reconstruction!")
    print(f"     Loading '{output_array_file}' directly...")
    print("     (Skipping the 6-minute GMM fitting process)\n")
    
    reconstructed_brain = np.load(output_array_file)

else:
    print(f"\n---> No saved reconstruction found. Fitting GMM from scratch...")
    mask = brain_data > threshold
    coordinates = np.argwhere(mask)
    actual_intensities = brain_data[mask]

    print(f"2. Fitting {n_components} components (Expected time: ~6 minutes)...")
    start_time = time.time()
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(coordinates, actual_intensities)
    print(f"   Done in {(time.time() - start_time)/60:.1f} minutes.")

    print("3. Reconstructing the 3D volume with Min-Max scaling...")
    pdf_values = np.exp(gmm.score_samples(coordinates))
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
    fig.suptitle(f"Original vs GMM ({n_components} Comp) - {view_name} View", fontsize=16)
    
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
            
        # Plot Original
        im_orig = ax_orig.imshow(slice_orig, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        ax_orig.set_title(f"Original Data (Slice {slice_idx})")
        ax_orig.axis('off')
        
        # Plot Reconstruction
        im_recon = ax_recon.imshow(slice_recon, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        ax_recon.set_title(f"GMM Reconstruction (Slice {slice_idx})")
        ax_recon.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(im_orig, cax=cbar_ax, orientation='horizontal', label='Voxel Intensity')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(filename, dpi=300)
    plt.close() 
    print(f"   -> Saved {filename}")

save_view_plot(axis=0, view_name="Sagittal", filename="gmm_recon_sagittal.png")
save_view_plot(axis=1, view_name="Coronal",  filename="gmm_recon_coronal.png")
save_view_plot(axis=2, view_name="Axial",    filename="gmm_recon_axial.png")

print("\nSuccess! Array saved and images generated.")