#!/bin/bash
#SBATCH --job-name=tierb
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100|a6000|a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/tierb_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/tierb_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of tierb.py (submit_tierb.sh chains them, each as its own job):
#   STAGE=select|labels|checks   CPU (submit with --partition=standard --gres= --constraint=)
#   STAGE=screen                 one GPU, WORKERS processes over the ranked candidates
#   STAGE=discover|verify        one GPU per group (array task = group), WORKERS processes
# Every stage skips work already written, so a stopped job can be resubmitted as is.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/hint_finetune/tierb}"
STAGE="${STAGE:?Pass STAGE=select|screen|discover|verify|labels|checks}"
WORKERS="${WORKERS:-8}"
EXTRA="${EXTRA:-}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"     # the gated facebook/UMA checkpoints
export OMP_NUM_THREADS=2
E=examples/MaterialsSaddles/experiments/hint_finetune/tierb.py
# Wait for every worker and fail the job if any worker failed (a bare `wait` returns 0 regardless).
wait_all() {
  local status=0
  for pid in "$@"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || { echo "a worker failed; see the .err file" >&2; exit 1; }
}
case "$STAGE" in
  select|labels|checks)
    "$PYTHON" -u "$E" "$STAGE" --out-dir "$OUT" $EXTRA ;;
  screen)
    nvidia-smi --query-gpu=name --format=csv,noheader
    pids=()
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" screen --out-dir "$OUT" --shard "$w" --nshards "$WORKERS" $EXTRA & pids+=($!)
    done
    wait_all "${pids[@]}" ;;
  discover|verify)
    GROUP="${GROUP:-${SLURM_ARRAY_TASK_ID:?array task id = group}}"
    nvidia-smi --query-gpu=name --format=csv,noheader
    pids=()
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" "$STAGE" --out-dir "$OUT" --group "$GROUP" --shard "$w" --nshards "$WORKERS" $EXTRA & pids+=($!)
    done
    wait_all "${pids[@]}" ;;
esac
