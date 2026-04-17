"""
Verify determinism: compare .npy outputs from two runs on the same subjects.
Reports bit-identical match, max absolute difference, and flags any mismatches.
"""

import os
import sys
import numpy as np

dir_a = 'VERIFICATION'
dir_b = 'gmm_features_weighted'

files_a = sorted([f for f in os.listdir(dir_a) if f.endswith('.npy')])
files_b = sorted([f for f in os.listdir(dir_b) if f.endswith('.npy')])

common = sorted(set(files_a) & set(files_b))
if not common:
    print(f"[!] No shared .npy files between {dir_a} and {dir_b}")
    sys.exit(1)

print(f"Comparing {len(common)} files\n")

all_identical = True
for fname in common:
    a = np.load(os.path.join(dir_a, fname))
    b = np.load(os.path.join(dir_b, fname))

    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH  {fname}: {a.shape} vs {b.shape}")
        all_identical = False
        continue

    bit_equal = np.array_equal(a, b)
    max_diff = np.max(np.abs(a - b))

    if bit_equal:
        print(f"  IDENTICAL       {fname}")
    else:
        all_identical = False
        rel_diff = max_diff / (np.max(np.abs(a)) + 1e-30)
        print(f"  MISMATCH        {fname}  max_abs_diff={max_diff:.2e}  max_rel_diff={rel_diff:.2e}")

print(f"\n{'ALL FILES BIT-IDENTICAL' if all_identical else 'SOME FILES DIFFER'}")
