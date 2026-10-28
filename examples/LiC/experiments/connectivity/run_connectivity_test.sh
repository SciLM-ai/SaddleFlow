#!/bin/bash
#SBATCH --job-name=lic_connectivity
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_connectivity_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_connectivity_%j.err
#SBATCH --no-requeue

# Downhill connectivity check of the never-reached coverage pairs. Extra arguments go to connectivity_test.py,
# e.g. --n-controls 5 (default) or --controls 0,1,2 to pick the control saddles by hand.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
R=/scratch/gqs7zr/saddleflow/lic_source_study_seed42/coverage_curve
OUT="${OUT:-$R/connectivity_${SLURM_JOB_ID:?Submit with sbatch}}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$PYTHON" -u examples/LiC/experiments/connectivity/connectivity_test.py \
  "$R/sigma0.15/source_study.npz" "$R/sigma0.30/source_study.npz" --out-dir "$OUT" "$@"
