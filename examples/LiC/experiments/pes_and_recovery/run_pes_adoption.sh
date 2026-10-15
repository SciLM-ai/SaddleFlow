#!/bin/bash
#SBATCH --job-name=lic_pes_adoption
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_pes_adoption_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_pes_adoption_%j.err

# Michael: existing-checkpoint diagnostic only. No training or optimizer.
set -euo pipefail
LIC_ADOPT_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_ADOPT_PYTHON="${LIC_ADOPT_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_ADOPT_RUN="${LIC_ADOPT_RUN:-/scratch/gqs7zr/saddleflow/lic_pes_directions_21242185}"
cd "$LIC_ADOPT_REPO"
mkdir -p "$LIC_ADOPT_RUN/fixed_probe_adoption"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LIC_ADOPT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_ADOPT_RUN/fixed_probe_adoption/matplotlib_cache"
echo "[batch] Michael: fixed-probe teacher adoption; run=$LIC_ADOPT_RUN; no training"
srun --unbuffered "$LIC_ADOPT_PYTHON" -u examples/LiC/experiments/pes_and_recovery/diagnose_pes_adoption.py \
    --run-dir "$LIC_ADOPT_RUN" --device cuda "$@"
