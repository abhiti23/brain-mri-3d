# Weighted GMM Feature Extraction

## Setup

```
parent_dir/
├── sub-*_preproc-cat12vbm_desc-gm_T1w.npy   ← subject MRIs
└── weighted_gmm/                              ← this directory
    ├── centers.npy
    ├── avg_full_model.npz
    └── fit_subjects_batch_numba.py
```

## Dependencies

```bash
pip install numpy scipy scikit-learn numba
```

## Run

```bash
cd weighted_gmm

# Test on 5 subjects
python fit_subjects_batch_numba.py --end 5 --workers 2

# Full batch
nohup python fit_subjects_batch_numba.py --workers 8 > run.log 2>&1 &
tail -f run.log
```

Set `--workers` to your available core count (`nproc`).

## Output

`gmm_features_weighted/` — one .npy per subject, 3500 features each.

Each feature vector: `[π_1×T, c_xx, c_xy, c_xz, c_yy, c_yz, c_zz, π_2×T, ...]` for 500 components.

## Loading features

```python
import numpy as np, glob
files = sorted(glob.glob('gmm_features_weighted/*.npy'))
X = np.array([np.load(f) for f in files])        # (n_subjects, 3500)
weights_only = X[:, 0::7]                         # (n_subjects, 500)
T = weights_only.sum(axis=1)                      # total GM volume per subject
```
