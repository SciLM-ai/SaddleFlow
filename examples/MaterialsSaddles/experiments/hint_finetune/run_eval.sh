#!/bin/bash
#SBATCH --job-name=hint_eval
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100|a6000|a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/hint_eval_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/hint_eval_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of eval_hints.py (submit_eval.sh chains them, each as its own job):
#   STAGE=land           one GPU per group (array task = group), one process; needs CKPT
#   STAGE=sella|verify   one GPU per group (array task = group), WORKERS processes
#   STAGE=report         CPU (submit with --partition=standard --gres= --constraint=)
# Every stage skips work already written, so a stopped job can be resubmitted as is.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:?Pass OUT=<evaluation directory>}"
STAGE="${STAGE:?Pass STAGE=land|sella|verify|report}"
WORKERS="${WORKERS:-8}"
EXTRA="${EXTRA:-}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"     # the gated facebook/UMA checkpoints
export OMP_NUM_THREADS=2
E=examples/MaterialsSaddles/experiments/hint_finetune/eval_hints.py
# Wait for every worker and fail the job if any worker failed (a bare `wait` returns 0 regardless).
wait_all() {
  local status=0
  for pid in "$@"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || { echo "a worker failed; see the .err file" >&2; exit 1; }
}
case "$STAGE" in
  report)
    "$PYTHON" -u "$E" report --out-dir "$OUT" $EXTRA ;;
  land)
    GROUP="${GROUP:-${SLURM_ARRAY_TASK_ID:?array task id = group}}"
    nvidia-smi --query-gpu=name --format=csv,noheader
    "$PYTHON" -u "$E" land --out-dir "$OUT" --group "$GROUP" --ckpt "${CKPT:?Pass CKPT=<checkpoint dir>}" $EXTRA ;;
  sella|verify)
    GROUP="${GROUP:-${SLURM_ARRAY_TASK_ID:?array task id = group}}"
    nvidia-smi --query-gpu=name --format=csv,noheader
    pids=()
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" "$STAGE" --out-dir "$OUT" --group "$GROUP" --shard "$w" --nshards "$WORKERS" $EXTRA & pids+=($!)
    done
    wait_all "${pids[@]}" ;;
esac
