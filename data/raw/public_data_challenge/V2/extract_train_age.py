# this py file extracts the brain age for each subject and stores them in
# VBM_extracted/train_y.npy
import numpy as np
import os
import pandas as pd

# ============================================
# CONFIG — paths assume this script is in V2/
# ============================================
DATA_DIR = ".."                         # parent folder
NPY_PATH = os.path.join(DATA_DIR, "train.tsv")
OUT_DIR  = os.path.join(DATA_DIR, "VBM_extracted")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
print("Loading data:", NPY_PATH)
subj_metadata = pd.read_csv(NPY_PATH, sep="\t")
N, D_total = subj_metadata.shape
print(f"Loaded: {N} subjects, {D_total} metadata per subject")

X_age = subj_metadata["age"]

# ============================================
# SAVE AGE DATA
# ============================================
out_path = os.path.join(OUT_DIR, f"train_y.npy")
np.save(out_path, X_age)
print(f"Done. Saved ages for {N} subjects to folder: {OUT_DIR}")
