Initial directory structure:

parent/
├── sub-0001_preproc-cat12vbm_desc-gm_T1w.npy    ← inputs, one level up
├── sub-0002_preproc-cat12vbm_desc-gm_T1w.npy
├── ...
└── fda_bspline/                                  ← working folder
    ├── fd_coef_worker.py
    ├── collect_coefs.py
    ├── run_fda.sh
    ├── run_fda_val.sh
    ├── collect_coefs.sh
    └── collect_coefs_val.sh

To extract B-spline coefficients for all subjects, run the SLURM array job from inside the fda_bspline/ folder:
sbatch run_fda.sh

This submits one job per subject. Each job fits a tensor-product cubic B-spline basis (n_basis=6, giving 6³=216 coefficients) to the 3D VBM image via least-squares projection. Jobs write individual coefficient files to train_vbm_coefs/.

Once all array jobs finish, stack the per-subject files into a single matrix:
python3 collect_coefs.py --coef_dir train_vbm_coefs --out artifacts/coefficients_final.npz
Repeat for the validation set using run_fda_val.sh and collect_coefs_val.sh.
Note: Requires 32G RAM per job. Edit --account and --mail-user in the SLURM scripts before submitting.

Final directory structure:

parent/
├── sub-0001_preproc-cat12vbm_desc-gm_T1w.npy    ← inputs, one level up
├── sub-0002_preproc-cat12vbm_desc-gm_T1w.npy
├── ...
└── fda_bspline/                                  ← working folder
    ├── fd_coef_worker.py
    ├── collect_coefs.py
    ├── run_fda.sh
    ├── run_fda_val.sh
    ├── collect_coefs.sh
    ├── collect_coefs_val.sh
    ├── train_vbm_coefs/                          ← per-subject .npy files (training)
    │   ├── coef_0000_sub-0001_...T1w.npy
    │   ├── coef_0001_sub-0002_...T1w.npy
    │   └── ...
    ├── val_vbm_coefs/                            ← per-subject .npy files (validation)
    │   └── ...
    └── artifacts/                                ← stacked outputs
        ├── coefficients_final.npz                ← training (3227, 216)
        ├── coefficients_final.csv
        ├── coefficients_final_subject_ids.txt
        ├── val_coefficients_final.npz            ← validation (757, 216)
        ├── val_coefficients_final.csv
        └── val_coefficients_final_subject_ids.txt
