#!/bin/bash
#SBATCH --job-name=dimer_det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/dimer_det_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/dimer_det_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of the dimer determinism experiment on one A100 80GB (submit_determinism.sh chains them).
#   STAGE=dimer      CONDITIONS[array task] (D0..D5), WORKERS processes over the starts
#   STAGE=cpu        CONDITION=D6|D7|D8|D9, one thread per process (submit with --partition=standard --gres= --constraint=)
#   STAGE=analyze    the report
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/dimer_determinism}"
STAGE="${STAGE:?Pass STAGE=dimer|cpu|analyze}"
WORKERS="${WORKERS:-2}"
EXTRA="${EXTRA:-}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"      # fixed thread count: BLAS results can depend on it
E=examples/MaterialsSaddles/experiments/dimer_determinism/dimer_determinism.py
CONDITIONS=(D0 D1 D2 D3 D4 D5)
case "$STAGE" in
  dimer)
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    C="${CONDITION:-${CONDITIONS[${SLURM_ARRAY_TASK_ID:-0}]}}"
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" dimer --out-dir "$OUT" --condition "$C" --shard "$w" --nshards "$WORKERS" $EXTRA &
    done
    wait ;;
  cpu)
    # The deterministic CPU conditions (D6..D9): one thread per process.
    C="${CONDITION:?Pass CONDITION=D6|D7|D8|D9}"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" dimer --out-dir "$OUT" --condition "$C" --shard "$w" --nshards "$WORKERS" $EXTRA &
    done
    wait ;;
  analyze)
    "$PYTHON" -u "$E" analyze --out-dir "$OUT" ;;
esac
