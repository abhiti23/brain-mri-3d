#!/bin/bash
# =============================================================================
# submit.sh — Helper to prepare and submit the GMM array job on Great Lakes
# Run this script from: /home/anvay/EECS545/
# Usage: bash submit.sh
# =============================================================================

# 1. Create log directory if it doesn't exist
mkdir -p /scratch/eecs545w26_class_root/eecs545w26_class/anvay/logs
echo "[*] Log directory ready: /home/anvay/EECS545/logs"

# 2. Confirm account name (prints your available accounts)
echo ""
echo "[*] Your available SLURM accounts (verify your account name below):"
my_accounts

# 3. Submit the job
cd /scratch/eecs545w26_class_root/eecs545w26_class/anvay/
echo ""
echo "[*] Submitting job array..."
sbatch run_gmm.sh

# 4. Show queue status
echo ""
echo "[*] Current job queue for anvay:"
squeue -u anvay
