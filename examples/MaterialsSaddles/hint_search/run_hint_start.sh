#!/bin/bash
#SBATCH --job-name=hint_start
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/hint_start_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/hint_start_%j.err
#SBATCH --no-requeue

# The released model from hint starts (hint_start.py) on one GPU: rollouts, Sella (WORKERS processes),
# verification of the new distinct points (mode_search.py verify), then the combined report.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:-/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/mp20bat_hint_search}"
WORKERS="${WORKERS:-4}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
D=examples/MaterialsSaddles/hint_search
"$PYTHON" -u $D/hint_start.py rollout --run-dir "$RUN" --out-dir "$OUT"
for w in $(seq 0 $((WORKERS - 1))); do
  "$PYTHON" -u $D/hint_start.py sella --run-dir "$RUN" --out-dir "$OUT" --shard "$w" --nshards "$WORKERS" &
done
wait
for w in $(seq 0 $((WORKERS - 1))); do
  "$PYTHON" -u $D/mode_search.py verify --run-dir "$RUN" --out-dir "$OUT" --tag hs --shard "$w" --nshards "$WORKERS" &
done
wait
"$PYTHON" -u $D/mode_search.py analyze --run-dir "$RUN" --out-dir "$OUT"
