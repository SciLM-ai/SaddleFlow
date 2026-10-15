#!/bin/bash
#SBATCH --job-name=lic_saddle_search
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_saddle_search_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_saddle_search_%j.err

# Michael: submit from the repository root:
#   sbatch examples/LiC/experiments/saddle_search/run_saddle_search.sh
#
# Optional extra Python arguments, for example a small execution check:
#   sbatch examples/LiC/experiments/saddle_search/run_saddle_search.sh --limit-starts 2 --restarts 1 \
#       --max-steps 8 --max-force-calls 80 --log-every-steps 1
#
# Override checkpoint/output/interpreter through these environment variables:
#   LIC_SEARCH_CKPT, LIC_SEARCH_OUT_DIR, LIC_SEARCH_PYTHON, SADDLEFLOW_REPO.
# The default output is NEW for each job. This is not a resumable runner.

set -euo pipefail

LIC_SEARCH_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_SEARCH_PYTHON="${LIC_SEARCH_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_SEARCH_CKPT="${LIC_SEARCH_CKPT:-/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final}"
LIC_SEARCH_OUT_DIR="${LIC_SEARCH_OUT_DIR:-/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_${SLURM_JOB_ID:?Submit this script with sbatch}}"

cd "$LIC_SEARCH_REPO"
mkdir -p "$LIC_SEARCH_OUT_DIR"

# Use the environment's Python directly: no interactive conda activation needed.
# Slurm sets CUDA_VISIBLE_DEVICES to the allocated GPU. Do NOT overwrite it.
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LIC_SEARCH_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_SEARCH_OUT_DIR/matplotlib_cache"

echo "[batch] job=${SLURM_JOB_ID} host=$(hostname) date=$(date --iso-8601=seconds)"
echo "[batch] python=$LIC_SEARCH_PYTHON"
echo "[batch] checkpoint=$LIC_SEARCH_CKPT"
echo "[batch] results=$LIC_SEARCH_OUT_DIR"
echo "[batch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

LIC_SEARCH_ARGS=(
    --ckpt "$LIC_SEARCH_CKPT"
    --out-dir "$LIC_SEARCH_OUT_DIR"
    --starts mixed
    --restarts 2
    --device cuda
    "$@"
)
printf '[batch] command:'
printf ' %q' "$LIC_SEARCH_PYTHON" -u examples/LiC/rollout_eval/compare_saddle_search.py "${LIC_SEARCH_ARGS[@]}"
printf '\n'

srun --unbuffered "$LIC_SEARCH_PYTHON" -u examples/LiC/rollout_eval/compare_saddle_search.py "${LIC_SEARCH_ARGS[@]}"
