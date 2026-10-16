#!/bin/bash
#SBATCH --job-name=lic_cond_infer
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_conditioning_infer_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_conditioning_infer_%j.err
#SBATCH --no-requeue

# Defaults reproduce the 48k-checkpoint exact-false-bridge check.
# Extra CLI arguments are forwarded; --ckpt/--out-dir/--K can override defaults.
set -euo pipefail
LIC_INFER_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_INFER_PYTHON="${LIC_INFER_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_INFER_CKPT="${LIC_INFER_CKPT:-/scratch/gqs7zr/saddleflow/lic_conditioning_21363449/checkpoint_step_00048000}"
LIC_INFER_OUT="${LIC_INFER_OUT:-/scratch/gqs7zr/saddleflow/lic_conditioning_21363449/false_exact_step48000}"
cd "$LIC_INFER_REPO"
mkdir -p "$LIC_INFER_OUT"
export PYTHONPATH="$LIC_INFER_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_INFER_OUT/matplotlib_cache"
# A binding inherited from an interactive job can name CPUs outside this job.
unset SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST
export SLURM_CPU_BIND=none
# Preserve Slurm's CUDA_VISIBLE_DEVICES instead of hard-coding a GPU index.
echo "[batch] job=${SLURM_JOB_ID:?Submit with sbatch}; checkpoint=$LIC_INFER_CKPT; default output=$LIC_INFER_OUT"
srun --cpu-bind=none --unbuffered "$LIC_INFER_PYTHON" -u examples/LiC/rollout_eval/force_guided_sampling.py \
  --ckpt "$LIC_INFER_CKPT" --out-dir "$LIC_INFER_OUT" --device cuda \
  --false-saddles-only --K 20 --batch-size 10 "$@"
