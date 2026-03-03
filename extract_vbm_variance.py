import numpy as np

# Load variance vector
var_train = np.load("var_train.npy")

# Modality lengths
VBM_LEN = 902629
SBM_LEN = 327684

# Extract VBM variance
vbm_var_flat = var_train[:VBM_LEN]

# Reshape
vbm_var_3d = vbm_var_flat.reshape(91, 109, 91)

# Save
np.save("var_vbm_3d.npy", vbm_var_3d)

print("Saved 3D VBM variance volume:", vbm_var_3d.shape)
