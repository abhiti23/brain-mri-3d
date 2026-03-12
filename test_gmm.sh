#!/bin/bash
#SBATCH --job-name=gmm_test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anvay@umich.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=00:30:00
#SBATCH --account=eecs545w26_class
#SBATCH --partition=standard
#SBATCH --output=/scratch/eecs545w26_class_root/eecs545w26_class/anvay/logs/test_%j.out
#SBATCH --error=/scratch/eecs545w26_class_root/eecs545w26_class/anvay/logs/test_%j.err

echo "Start: $(date)"

module load python3.11-anaconda/2024.02

cd /scratch/eecs545w26_class_root/eecs545w26_class/anvay/

# Test 8 unprocessed subjects with 4 workers
python get_gaussian_params_v2.py --start 172 --end 180 --workers 4

echo "End: $(date)"
