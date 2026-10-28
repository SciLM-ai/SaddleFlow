#!/bin/bash
#SBATCH --job-name=hint_search
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/hint_search_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/hint_search_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of the mp20bat hint search; submit_hint_search.sh chains them. MODE is one of
#   model   released-model rollouts (one GPU)
#   search  Sella from every start (array task = GPU; WORKERS processes share each GPU)
#   verify  index and connectivity of every distinct point (array task = GPU; WORKERS per GPU)
#   analyze the report (CPU)
# LIMIT=N restricts every stage to the first N selected minima (the cost check).
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:?Pass RUN=<failure-mode run dir>}"
OUT="${OUT:?Pass OUT=<output dir>}"
MODE="${MODE:?Pass MODE=model|search|verify|analyze}"
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-0}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
export MPLCONFIGDIR="$OUT/matplotlib_cache"
SEARCH=(examples/MaterialsSaddles/hint_search/hint_search.py)
COMMON=(--run-dir "$RUN" --out-dir "$OUT" --limit "$LIMIT")
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS=$(( ${SLURM_ARRAY_TASK_COUNT:-1} * WORKERS ))
case "$MODE" in
  search|verify)
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "${SEARCH[@]}" "$MODE" "${COMMON[@]}" --shard $(( TASK * WORKERS + w )) --nshards "$NSHARDS" &
    done
    wait ;;
  model|analyze)
    "$PYTHON" -u "${SEARCH[@]}" "$MODE" "${COMMON[@]}" ;;
esac
