import numpy as np

# Load mean variance or any sample to inspect structure
mean_train = np.load("mean_train.npy")

# Known modality lengths from OpenBHB
VBM_LEN = 902629
SBM_LEN = 327684
QUASI_LEN = 2429259  # remainder

# Extract slices
vbm_flat = mean_train[:VBM_LEN]
sbm_flat = mean_train[VBM_LEN:VBM_LEN+SBM_LEN]
quasi_flat = mean_train[VBM_LEN+SBM_LEN:]

print("VBM:", vbm_flat.shape)
print("SBM:", sbm_flat.shape)
print("QuasiRaw:", quasi_flat.shape)

# Reshape VBM into a real 3D MRI
vbm_3d = vbm_flat.reshape(91, 109, 91)

np.save("mean_vbm_3d.npy", vbm_3d)
print("Saved 3D VBM volume.")
