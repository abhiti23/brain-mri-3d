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


Run the commands in the following order:

1. sbatch run_fda.sh — SLURM array job that extracts B-spline coefficients for all training subjects (one job per subject), outputs stored in train_vbm_coefs/
2. sbatch --dependency=afterok:<JOB_ID> collect_coefs.sh — Stacks all training coefficient files into a single matrix, outputs stored in artifacts/
3. sbatch run_fda_val.sh — SLURM array job that extracts B-spline coefficients for all validation subjects, outputs stored in val_vbm_coefs/
4. sbatch --dependency=afterok:<JOB_ID> collect_coefs_val.sh — Stacks all validation coefficient files into a single matrix, outputs stored in artifacts/


Files and what they do:
1. fd_coef_worker.py — The core worker script. Takes a single subject index as input, loads the corresponding VBM .npy file, squeezes it from (1, 1, 121, 145, 121) to (121, 145, 121), fits a tensor-product cubic B-spline basis via least-squares, and saves the resulting 216-dimensional coefficient vector as a .npy file. Called once per subject by the SLURM array jobs.
2. collect_coefs.py — Collector script. Scans a directory of per-subject coefficient .npy files, checks for missing or corrupted files, stacks them into a single (n_subjects, 216) matrix, and saves the result as .npz, .csv, and a subject ID list for label alignment.
3. run_fda.sh — SLURM array job script for the training set. Submits one job per training subject (3227 total, --array=0-3226), each calling fd_coef_worker.py with the appropriate subject index and paths. Allocates 32G RAM and 30 minutes per job.
4. run_fda_val.sh — Same as run_fda.sh but for the validation set (757 subjects, --array=0-756). Points to val_vbm/ as input and val_vbm_coefs/ as output.
5. collect_coefs.sh — SLURM job that runs collect_coefs.py for the training set. Typically submitted with --dependency=afterok:<JOB_ID> so it triggers automatically once all training array jobs complete.
6. collect_coefs_val.sh — Same as collect_coefs.sh but for the validation set. Reads from val_vbm_coefs/ and writes to artifacts/val_coefficients_final.npz.
