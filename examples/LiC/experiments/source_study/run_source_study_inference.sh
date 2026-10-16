#!/bin/bash
#SBATCH --job-name=lic_source_eval
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_source_eval_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_source_eval_%j.err
#SBATCH --no-requeue

set -euo pipefail
LIC_STUDY_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_STUDY_PYTHON="${LIC_STUDY_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_STUDY_ROOT="${LIC_STUDY_ROOT:-/scratch/gqs7zr/saddleflow/lic_source_study_seed42}"
cd "$LIC_STUDY_REPO"
export PYTHONPATH="$LIC_STUDY_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/lic_source_eval_${SLURM_JOB_ID:?Submit with sbatch}"
unset SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST
export SLURM_CPU_BIND=none
echo "[batch] job=$SLURM_JOB_ID source-study inference; root=$LIC_STUDY_ROOT"
srun --cpu-bind=none --unbuffered "$LIC_STUDY_PYTHON" -u examples/LiC/experiments/source_study/source_study.py compare \
  --study-dir "$LIC_STUDY_ROOT" --device cuda "$@"
