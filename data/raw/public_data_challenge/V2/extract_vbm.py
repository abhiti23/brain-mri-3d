import numpy as np
import os
import nibabel as nib
from nilearn.maskers import NiftiMasker

# ============================================
# CONFIG — paths assume this script is in V2/
# ============================================
DATA_DIR = ".."                         # parent folder
NPY_PATH = os.path.join(DATA_DIR, "train.npy")
# the following file stores details about the masking/atlas
MASK_PATH = "../cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
OUT_DIR  = os.path.join(DATA_DIR, "VBM_extracted")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# INITIALIZE MASKER (The "Instruction Manual")
# ============================================
print(f"Loading mask: {MASK_PATH}")
mask_img = nib.load(MASK_PATH)

# We create a binary mask (True where gray matter probability > 0.05)
# This ensures we match the OpenBHB compression exactly
mask_data = mask_img.get_fdata() > 0.05
binary_mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), mask_img.affine)

# Initialize the tool that converts 1D -> 3D
masker = NiftiMasker(mask_img=binary_mask_img).fit()
VBM_SIZE = np.sum(mask_data)
print(f"Correct VBM block size identified: {VBM_SIZE} voxels")


# ============================================
# LOAD DATA
# ============================================
print("Loading data:", NPY_PATH)
X = np.load(NPY_PATH, mmap_mode="r")
N, D_total = X.shape
print(f"Loaded: {N} subjects, {D_total} features per subject")

# ============================================
# PROCESS AND UNMASK SUBJECTS
# ============================================
print(f"Extracting VBM indices 0-{VBM_SIZE} and unmasking to 3D...")

for i in range(N):
    # 1. Extract the 1D masked VBM data for this subject
    subject_vbm_1d = X[i, 0:VBM_SIZE]

    # 2. Use the masker to put the values back into 3D space
    # This replaces the broken .reshape() command
    subject_vbm_3d_img = masker.inverse_transform(subject_vbm_1d)

    # 3. Convert to a standard numpy array (Shape will be 121x145x121)
    vbm_3d_volume = subject_vbm_3d_img.get_fdata()

    # 4. Save per-subject file as requested
    out_path = os.path.join(OUT_DIR, f"VBM_{i}.npy")
    np.save(out_path, vbm_3d_volume)

    if i % 100 == 0:
        print(f"Progress: {i}/{N} subjects processed...")

print(f"Done. Saved {N} corrected VBM volumes to folder: {OUT_DIR}")