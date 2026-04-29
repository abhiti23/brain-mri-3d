#!/bin/bash
# =============================================================================
# run_fda_val.sh  —  SLURM array job for VALIDATION VBM coefficient extraction
#
# Usage:
#   sbatch run_fda_val.sh              # full validation run
#   sbatch --array=0-4 run_fda_val.sh  # test run
# =============================================================================

#SBATCH --job-name=vbm_fda_val
#SBATCH --array=0-756            # 757 validation subjects
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=eecs545w26_class   # <-- confirm with: my_accounts
#SBATCH --partition=standard
#SBATCH --output=logs/vbm_val_%A_%a.out
#SBATCH --error=logs/vbm_val_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_UMICH_EMAIL@umich.edu   # <-- replace

# --- Paths -------------------------------------------------------------------
BASE="/scratch/eecs545w26_class_root/eecs545w26_class/anvay"
VBM_DIR="$BASE/validation/val_vbm"
COEF_DIR="$BASE/validation/val_vbm_coefs"
LOG_DIR="$BASE/logs"

mkdir -p "$COEF_DIR" "$LOG_DIR"

# --- Environment -------------------------------------------------------------
module load python/3.11

# --- Run ---------------------------------------------------------------------
echo "Val subject $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

python "$BASE/fd_coef_worker.py" \
    "$SLURM_ARRAY_TASK_ID" \
    --n_basis 6 \
    --data_dir "$VBM_DIR" \
    --out_dir "$COEF_DIR"

EXIT_CODE=$?
echo "Finished $SLURM_ARRAY_TASK_ID with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE
