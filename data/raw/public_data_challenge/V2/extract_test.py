# this file extracts test data (vbm and brain age) and stores them in data/raw/public_data_challenge/VBM_extracted 
import numpy as np
import os
import pandas as pd

# ============================================
# CONFIG — paths assume this script is in V2/
# ============================================
DATA_DIR = ".."                         # parent folder
OUT_DIR  = os.path.join(DATA_DIR, "VBM_extracted")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# LOAD TEST VBM DATA
# ============================================
NPY_PATH = os.path.join(DATA_DIR, "test.npy")
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
    out_path = os.path.join(OUT_DIR, f"VBM_test_{i}.npy")
    np.save(out_path, X_VBM_3D[i])

print(f"Done. Saved {N} VBM volumes to folder: {OUT_DIR}")


# ============================================
# LOAD TEST AGE DATA
# ============================================
NPY_PATH = os.path.join(DATA_DIR, "test.tsv")
print("Loading data:", NPY_PATH)
subj_metadata = pd.read_csv(NPY_PATH, sep="\t")
N, D_total = subj_metadata.shape
print(f"Loaded: {N} subjects, {D_total} features per subject")

X_age = subj_metadata["age"]

# ============================================
# SAVE TEST AGE DATA
# ============================================
out_path = os.path.join(OUT_DIR, f"test_y.npy")
np.save(out_path, X_age)
print(f"Done. Saved {N} ages to folder: {OUT_DIR}")