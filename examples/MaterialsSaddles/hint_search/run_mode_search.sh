#!/bin/bash
#SBATCH --job-name=mode_search
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/mode_search_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/mode_search_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of mode_search.py on the hint-search output: MODE=search|verify (array task = GPU, WORKERS
# processes per GPU) or MODE=analyze (CPU). submit_mode_search.sh chains them.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:-/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/mp20bat_hint_search}"
MODE="${MODE:?Pass MODE=search|verify|analyze}"
WORKERS="${WORKERS:-4}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
M=examples/MaterialsSaddles/hint_search/mode_search.py
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS=$(( ${SLURM_ARRAY_TASK_COUNT:-1} * WORKERS ))
if [ "$MODE" = analyze ]; then
  "$PYTHON" -u "$M" analyze --run-dir "$RUN" --out-dir "$OUT"
else
  for w in $(seq 0 $((WORKERS - 1))); do
    "$PYTHON" -u "$M" "$MODE" --run-dir "$RUN" --out-dir "$OUT" --shard $(( TASK * WORKERS + w )) --nshards "$NSHARDS" &
  done
  wait
fi
