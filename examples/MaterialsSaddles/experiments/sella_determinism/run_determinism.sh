#!/bin/bash
#SBATCH --job-name=sella_det
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/sella_det_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/sella_det_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of the Sella determinism experiment on one A100 80GB (submit_determinism.sh chains them).
#   STAGE=forces            tier 1 on the GPU: gpu_default, gpu_seeded, gpu_det_seeded, each in its own process
#   STAGE=sella             tier 2: CONDITIONS[array task] (C0..C5), WORKERS processes over the starts
#   STAGE=forces_cpu|cpu    the CPU parts (cpu: CONDITION=C6|C7|C8, one thread per process) (submit with --partition=standard --gres= --constraint=)
#   STAGE=analyze           the report
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/sella_determinism}"
STAGE="${STAGE:?Pass STAGE=forces|sella|forces_cpu|cpu|analyze}"
WORKERS="${WORKERS:-2}"
EXTRA="${EXTRA:-}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"      # fixed thread count: BLAS results can depend on it
E=examples/MaterialsSaddles/experiments/sella_determinism/sella_determinism.py
CONDITIONS=(C0 C1 C2 C3 C4 C5)
case "$STAGE" in
  forces)
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    for c in gpu_default gpu_seeded gpu_det_seeded; do
      "$PYTHON" -u "$E" forces --out-dir "$OUT" --config "$c" $EXTRA
    done ;;
  forces_cpu)
    for c in cpu cpu_seeded cpu_1thread_seeded; do
      "$PYTHON" -u "$E" forces --out-dir "$OUT" --config "$c" $EXTRA
    done ;;
  sella)
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    C=${CONDITIONS[${SLURM_ARRAY_TASK_ID:-0}]}
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" sella --out-dir "$OUT" --condition "$C" --shard "$w" --nshards "$WORKERS" $EXTRA &
    done
    wait ;;
  cpu)
    # The deterministic CPU conditions (C6, C7, C8): one thread per process.
    C="${CONDITION:?Pass CONDITION=C6|C7|C8}"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$E" sella --out-dir "$OUT" --condition "$C" --shard "$w" --nshards "$WORKERS" $EXTRA &
    done
    wait ;;
  analyze)
    "$PYTHON" -u "$E" analyze --out-dir "$OUT" ;;
esac
