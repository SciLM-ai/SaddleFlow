#!/bin/bash
#SBATCH --job-name=higher_order
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/higher_order_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/higher_order_%x_%A_%a.err
#SBATCH --no-requeue

# Averaging and integration test of the higher-order landing points (higher_order_test.py).
#   RUN=<failure-mode run dir>
#   sbatch --array=0-7 --export=ALL,MODE=neighbours,RUN=$RUN run_higher_order.sh   # Sella, 8 shards
#   sbatch --export=ALL,MODE=rollouts,RUN=$RUN run_higher_order.sh                  # model rollouts
#   sbatch --dependency=afterany:<ids> --partition=standard --gres=none --constraint= --time=00:30:00 \
#          --export=ALL,MODE=analyze,RUN=$RUN run_higher_order.sh                   # report, CPU
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:?Pass RUN=<failure-mode run dir>}"
MODE="${MODE:?Pass MODE=neighbours|rollouts|analyze}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$RUN/matplotlib_cache"
SHARD=()
if [ "$MODE" = neighbours ]; then
  SHARD=(--shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${SLURM_ARRAY_TASK_COUNT:-1}")
fi
"$PYTHON" -u examples/MaterialsSaddles/failure_mode_analysis/higher_order_test.py "$MODE" "$RUN" "${SHARD[@]}" "$@"
