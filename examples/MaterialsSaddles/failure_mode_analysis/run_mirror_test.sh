#!/bin/bash
#SBATCH --job-name=mp20_mirror
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/mp20_mirror_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/mp20_mirror_%j.err
#SBATCH --no-requeue

# Inversion-equivariance test of the released model on a failure-mode run (mirror_test.py).
#   sbatch --export=ALL,RUN=<failure-mode run dir> run_mirror_test.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:-/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PYTHON" -u examples/MaterialsSaddles/failure_mode_analysis/mirror_test.py "$RUN" "$@"
