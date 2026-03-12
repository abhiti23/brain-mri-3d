#!/bin/bash
# =============================================================================
# SLURM Array Job: Fixed-Means GMM on OpenBHB VBM Data
# Great Lakes HPC - University of Michigan
#
# Strategy: Split 3,227 subjects across SLURM array tasks.
#   - CHUNK_SIZE controls how many subjects each task handles.
#   - With CHUNK_SIZE=100, we need ceil(3227/100) = 33 array tasks.
#   - Each task runs ~100 subjects × ~3 mins = ~5 hrs worst case.
#   - Adjust CHUNK_SIZE and --time if your per-subject time differs.
# =============================================================================

#SBATCH --job-name=gmm_vbm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anvay@umich.edu

# --- Array: tasks 0 through 32 (33 tasks × 100 subjects = 3,300 slots, covers 3,227) ---
#SBATCH --array=0-32

# --- Resources per task ---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # Single process per task (GMM is not MPI)
#SBATCH --cpus-per-task=4          # sklearn can use multiple CPUs via OpenBLAS
#SBATCH --mem-per-cpu=8gb          # 8gb × 4 CPUs = 32gb per task; GMM on 3D coords needs room
#SBATCH --time=06:00:00            # 6 hrs per task (safe buffer for 100 subjects × ~3 min each)

# --- Account & Partition ---
#SBATCH --account=eecs545w26_class
#SBATCH --partition=standard

# --- Logging: one log file per array task ---
#SBATCH --output=/scratch/eecs545w26_class_root/eecs545w26_class/anvay/logs/gmm_%A_%a.out
#SBATCH --error=/scratch/eecs545w26_class_root/eecs545w26_class/anvay/logs/gmm_%A_%a.err

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "------------------------------------------------------------"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Start time:    $(date)"
echo "------------------------------------------------------------"

# Load a Python module available on Great Lakes
module load python3.11-anaconda/2024.02

# Allow sklearn/numpy to use the CPUs we requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =============================================================================
# COMPUTE START/END INDICES FOR THIS ARRAY TASK
# =============================================================================

CHUNK_SIZE=100
TASK_ID=$SLURM_ARRAY_TASK_ID

START=$(( TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))

echo "This task will process subjects index $START to $END"

# =============================================================================
# RUN THE GMM SCRIPT
# =============================================================================

cd /scratch/eecs545w26_class_root/eecs545w26_class/anvay/

python get_gaussian_params.py --start $START --end $END

echo "------------------------------------------------------------"
echo "End time: $(date)"
echo "------------------------------------------------------------"
