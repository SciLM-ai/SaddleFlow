#!/bin/bash
#SBATCH --job-name=lic_pes_directions
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_pes_directions_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_pes_directions_%j.err
#SBATCH --no-requeue

# Sbatch examples/LiC/experiments/pes_and_recovery/run_pes_directions.sh
# Runs calibration, matched E0/E1/E2 continuation, then unguided comparison.
# For interrupted training, use the standalone trainer's --resume-from.
set -euo pipefail
LIC_PES_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_PES_PYTHON="${LIC_PES_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_PES_OUT="${LIC_PES_OUT:-/scratch/gqs7zr/saddleflow/lic_pes_directions_${SLURM_JOB_ID:?Submit with sbatch}}"
LIC_PES_INIT="${LIC_PES_INIT:-/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final}"
LIC_PES_STARTS="${LIC_PES_STARTS:-/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_21107961/saddle_search.npz}"
cd "$LIC_PES_REPO"
mkdir -p "$LIC_PES_OUT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$LIC_PES_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_PES_OUT/matplotlib_cache"
echo "[batch] job=$SLURM_JOB_ID results=$LIC_PES_OUT"
LIC_PES_SHARED=(--init-weights "$LIC_PES_INIT" --calibration-dir "$LIC_PES_OUT/calibration" --device cuda)
srun --unbuffered "$LIC_PES_PYTHON" -u examples/LiC/experiments/pes_and_recovery/train_pes_directions.py "${LIC_PES_SHARED[@]}" --calibrate-only
for LIC_PES_EXPERIMENT in E0 E1 E2; do
    srun --unbuffered "$LIC_PES_PYTHON" -u examples/LiC/experiments/pes_and_recovery/train_pes_directions.py "${LIC_PES_SHARED[@]}" \
        --experiment "$LIC_PES_EXPERIMENT" --output-dir "$LIC_PES_OUT/$LIC_PES_EXPERIMENT"
done
srun --unbuffered "$LIC_PES_PYTHON" -u examples/LiC/experiments/pes_and_recovery/compare_pes_directions.py \
    --run-dir "$LIC_PES_OUT" --out-dir "$LIC_PES_OUT/comparison" --starts-archive "$LIC_PES_STARTS" --device cuda
