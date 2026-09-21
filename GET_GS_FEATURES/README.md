# GET_GS_FEATURES

## Initial directory structure

```
parent/
├── sub-0001_preproc-cat12vbm_desc-gm_T1w.npy    ← inputs, one level up
├── sub-0002_preproc-cat12vbm_desc-gm_T1w.npy
├── ...
└── weighted_gmm/                                 ← working folder
    ├── group_average_gm_T1w.npy
    ├── fit_average_brain_free_means.npy
    ├── fit_subjects_batch_numba.py
    └── reconstruct_from_saved_model.py
```

The parent directory contains the individual VBM `.npy` files. Each such file has size of 17MB. The average brain (`group_average_gm_T1w.npy`) can be obtained from a simple script that takes a random sample of brains (we chose a random sample of size 150) and calculates the arithmetic mean.

## Commands

To run the full model (finding the centers, intensities, and covariances) on the average brain to get the parameters (in the files `centers.npy` and `avg_model.npz`), run the script:

```bash
python fit_average_brain_free_means.py group_average_gm_T1w.npy --out centers.npy --save-full avg_model.npz
```

To obtain the average brain reconstructed using the extracted parameters, run the script:

```bash
python reconstruct_from_saved_model.py avg_model.npz group_average_gm_T1w.npy --save-recon avg_reconstructed.npy
```

This produces the file `avg_reconstructed.npy`, and the reconstruction can be visualized by several plotting scripts which we do not include here.

Finally, to get the features for individual subjects (while the centers are frozen, intensities and covariances initialized from the values obtained on the average brain), run the script:

```bash
python fit_subjects_batch_numba.py --workers 8
```

If running on the cluster, set the number of workers to be the number of cores available.

## Final directory structure

```
parent/
├── sub-0001_preproc-cat12vbm_desc-gm_T1w.npy    ← inputs, one level up
├── sub-0002_preproc-cat12vbm_desc-gm_T1w.npy
├── ...
└── weighted_gmm/                                 ← working folder
    ├── group_average_gm_T1w.npy
    ├── centers.npy
    ├── avg_model.npz
    ├── avg_reconstructed.npy
    ├── fit_average_brain_free_means.npy
    ├── fit_subjects_batch_numba.py
    ├── reconstruct_from_saved_model.py
    └── gmm_features_weighted/                    ← outputs land here
        ├── sub-0001_preproc-cat12vbm_desc-gm_T1w.npy
        ├── sub-0002_preproc-cat12vbm_desc-gm_T1w.npy
        └── ...
```
