#!/bin/bash
#SBATCH --job-name=lic_mmf_mobility
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_mmf_mobility_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_mmf_mobility_%j.err
#SBATCH --no-requeue

# Sbatch examples/LiC/experiments/mmf_mobility/run_mmf_mobility.sh
# Additional arguments are training arguments, e.g. --num-updates 400.
# For resume, set LIC_MMF_OUT_DIR to the original run and pass --resume.
# The finished historical baseline supplies identical starts and Sella/FM rows.
# Set LIC_MMF_BASELINE to an empty string to generate fresh mixed starts instead.

set -euo pipefail

LIC_MMF_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_MMF_PYTHON="${LIC_MMF_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_MMF_OUT_DIR="${LIC_MMF_OUT_DIR:-/scratch/gqs7zr/saddleflow/lic_mmf_mobility_${SLURM_JOB_ID:?Submit with sbatch}}"
LIC_MMF_BASELINE="${LIC_MMF_BASELINE-/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_21107961/saddle_search.npz}"

cd "$LIC_MMF_REPO"
mkdir -p "$LIC_MMF_OUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LIC_MMF_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_MMF_OUT_DIR/matplotlib_cache"

echo "[batch] job=${SLURM_JOB_ID} results=$LIC_MMF_OUT_DIR"
echo "[batch] training global/state policies; Kmax=100 by default"
echo "[batch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "[batch] baseline=$LIC_MMF_BASELINE"

srun --unbuffered "$LIC_MMF_PYTHON" -u examples/LiC/experiments/mmf_mobility/train_mmf_mobility.py \
    --out-dir "$LIC_MMF_OUT_DIR" --max-steps 100 --device cuda "$@"

LIC_MMF_COMPARE_ARGS=(--run-dir "$LIC_MMF_OUT_DIR" --out-dir "$LIC_MMF_OUT_DIR/comparison_${SLURM_JOB_ID}" --device cuda)
if [[ -n "$LIC_MMF_BASELINE" ]]; then
    LIC_MMF_COMPARE_ARGS+=(--baseline-archive "$LIC_MMF_BASELINE")
fi
srun --unbuffered "$LIC_MMF_PYTHON" -u examples/LiC/experiments/mmf_mobility/compare_mmf_mobility.py "${LIC_MMF_COMPARE_ARGS[@]}"
