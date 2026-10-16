#!/bin/bash
#SBATCH --job-name=lic_conditioning
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_conditioning_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_conditioning_%j.err
#SBATCH --no-requeue

# Flags opt in; no flags runs the ordinary baseline recipe.
# sbatch examples/LiC/experiments/force_conditioning/run_conditioning.sh --inject-force --curvature-conditioning
set -euo pipefail
LIC_COND_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_COND_PYTHON="${LIC_COND_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_COND_OUT="${LIC_COND_OUT:-/scratch/gqs7zr/saddleflow/lic_conditioning_${SLURM_JOB_ID:?Submit with sbatch}}"
cd "$LIC_COND_REPO"
mkdir -p "$LIC_COND_OUT"
export PYTHONPATH="$LIC_COND_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_COND_OUT/matplotlib_cache"
echo "[batch] job=$SLURM_JOB_ID; output=$LIC_COND_OUT; baseline settings unless CLI overrides them"
srun --unbuffered "$LIC_COND_PYTHON" -u examples/LiC/train.py --output-dir "$LIC_COND_OUT" --device cuda "$@"
