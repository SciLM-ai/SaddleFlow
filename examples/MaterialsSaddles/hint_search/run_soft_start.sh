#!/bin/bash
#SBATCH --job-name=soft_start
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/soft_start_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/soft_start_%j.err
#SBATCH --no-requeue

# Soft-mode starts on the two hint-search minima (hint_start.py soft_*): rollouts and null copies, Sella
# (WORKERS processes), verification of the new distinct points (mode_search.py verify), then both reports.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:-/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/mp20bat_hint_search}"
WORKERS="${WORKERS:-8}"
EXTRA="${EXTRA:-}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
D=examples/MaterialsSaddles/hint_search
nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" -u $D/hint_start.py soft_rollout --run-dir "$RUN" --out-dir "$OUT" $EXTRA
for w in $(seq 0 $((WORKERS - 1))); do
  "$PYTHON" -u $D/hint_start.py soft_sella --run-dir "$RUN" --out-dir "$OUT" --shard "$w" --nshards "$WORKERS" $EXTRA &
done
wait
for w in $(seq 0 $((WORKERS - 1))); do
  "$PYTHON" -u $D/mode_search.py verify --run-dir "$RUN" --out-dir "$OUT" --tag soft --shard "$w" --nshards "$WORKERS" &
done
wait
"$PYTHON" -u $D/hint_start.py soft_report --run-dir "$RUN" --out-dir "$OUT" $EXTRA
"$PYTHON" -u $D/mode_search.py analyze --run-dir "$RUN" --out-dir "$OUT"
