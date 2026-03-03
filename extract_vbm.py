import numpy as np
import os

# ============================================
# CONFIG — paths assume this script is in V2/
# ============================================
DATA_DIR = ".."                         # parent folder
NPY_PATH = os.path.join(DATA_DIR, "train.npy")
OUT_DIR  = os.path.join(DATA_DIR, "VBM_extracted")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
print("Loading data:", NPY_PATH)
X = np.load(NPY_PATH, mmap_mode="r")
N, D_total = X.shape
print(f"Loaded: {N} subjects, {D_total} features per subject")

# ============================================
# EXTRACT VBM BASED ON KNOWN VBM SIZE
# ============================================
VBM_SIZE = 91 * 109 * 91  # 902,629 voxels per subject
vbm_start = 0              # VBM is always first in OpenBHB concatenation
vbm_end = vbm_start + VBM_SIZE

if vbm_end > D_total:
    raise RuntimeError(
        f"VBM block ({vbm_start}-{vbm_end}) exceeds total features ({D_total})."
    )

print(f"Extracting VBM indices {vbm_start}-{vbm_end} (length={VBM_SIZE})")
X_VBM = X[:, vbm_start:vbm_end]

# ============================================
# RESHAPE TO 3D
# ============================================
VBM_SHAPE = (91, 109, 91)
X_VBM_3D = X_VBM.reshape((N,) + VBM_SHAPE)
print(f"Reshaped VBM volumes to 3D: {VBM_SHAPE}")

# ============================================
# SAVE PER-SUBJECT FILES
# ============================================
print("Saving per-subject VBM volumes…")
for i in range(N):
    out_path = os.path.join(OUT_DIR, f"VBM_{i}.npy")
    np.save(out_path, X_VBM_3D[i])

print(f"Done. Saved {N} VBM volumes to folder: {OUT_DIR}")