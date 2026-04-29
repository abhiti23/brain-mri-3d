#!/bin/bash
# =============================================================================
# collect_coefs.sh  —  Stack all per-subject coef files into final matrix
#
# Run manually:
#   sbatch collect_coefs.sh
#
# Or auto-chain after array job:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> collect_coefs.sh
# =============================================================================

#SBATCH --job-name=vbm_collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --account=eecs545w26_class   # <-- confirm with: my_accounts
#SBATCH --partition=standard
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_UMICH_EMAIL@umich.edu   # <-- replace

BASE="/scratch/eecs545w26_class_root/eecs545w26_class/anvay"
COEF_DIR="$BASE/training/train_vbm_coefs"
OUT_DIR="$BASE/artifacts"

mkdir -p "$OUT_DIR"

module load python/3.11

echo "Collecting at $(date)"

python "$BASE/collect_coefs.py" \
    --coef_dir "$COEF_DIR" \
    --out "$OUT_DIR/coefficients_final.npz"

echo "Done at $(date)"
