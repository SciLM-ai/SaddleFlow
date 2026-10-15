#!/bin/bash
#SBATCH --job-name=lic_kappa_audit
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_kappa_audit_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_kappa_audit_%j.err

# Michael: audit ALL stored paths; publish partial CPU-plottable snapshots.
set -euo pipefail
LIC_KAPPA_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_KAPPA_PYTHON="${LIC_KAPPA_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_KAPPA_OUT="${LIC_KAPPA_OUT:-/scratch/gqs7zr/saddleflow/lic_kappa_audit_${SLURM_JOB_ID:?Submit with sbatch}}"
LIC_KAPPA_SHELLS="${LIC_KAPPA_SHELLS:-/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz}"
LIC_KAPPA_SEARCH="${LIC_KAPPA_SEARCH:-/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_21107961/saddle_search.npz}"
cd "$LIC_KAPPA_REPO"
mkdir -p "$LIC_KAPPA_OUT"
export PYTHONPATH="$LIC_KAPPA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_KAPPA_OUT/matplotlib_cache"
echo "[batch] Michael: job=$SLURM_JOB_ID output=$LIC_KAPPA_OUT; no training or steering"
srun --unbuffered "$LIC_KAPPA_PYTHON" -u examples/LiC/experiments/kappa_audit/kappa_audit.py \
    --archives "$LIC_KAPPA_SHELLS" "$LIC_KAPPA_SEARCH" --out-dir "$LIC_KAPPA_OUT" --device cuda \
    --save-every-chunks 8 "$@"
srun --unbuffered "$LIC_KAPPA_PYTHON" -u examples/LiC/experiments/kappa_audit/plot_kappa_audit.py "$LIC_KAPPA_OUT/kappa_audit.npz"
