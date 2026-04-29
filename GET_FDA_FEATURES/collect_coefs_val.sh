#!/bin/bash
# =============================================================================
# collect_coefs_val.sh  —  Stack validation coefficients into final matrix
#
# Run manually:   sbatch collect_coefs_val.sh
# Auto-chain:     sbatch --dependency=afterok:<VAL_JOB_ID> collect_coefs_val.sh
# =============================================================================

#SBATCH --job-name=vbm_collect_val
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --account=eecs545w26_class
#SBATCH --partition=standard
#SBATCH --output=logs/collect_val_%j.out
#SBATCH --error=logs/collect_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_UMICH_EMAIL@umich.edu   # <-- replace

BASE="/scratch/eecs545w26_class_root/eecs545w26_class/anvay"
COEF_DIR="$BASE/validation/val_vbm_coefs"
OUT_DIR="$BASE/artifacts"

mkdir -p "$OUT_DIR"

module load python/3.11

echo "Collecting validation coefs at $(date)"

python "$BASE/collect_coefs.py" \
    --coef_dir "$COEF_DIR" \
    --out "$OUT_DIR/val_coefficients_final.npz"

echo "Done at $(date)"
