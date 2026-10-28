#!/bin/bash
#SBATCH --job-name=hvn
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/hint_vs_noise_%x_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/hint_vs_noise_%x_%j.err
#SBATCH --no-requeue

# One group of minima of hint_vs_noise.py on one GPU: rollouts, then Sella and verification with WORKERS
# processes each. STAGE=analyze writes the report instead (submit with --partition=standard --gres= --constraint=).
# Every stage skips work already written, so a job that stopped can be resubmitted as is.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/hint_vs_noise}"
STAGE="${STAGE:-group}"
WORKERS="${WORKERS:-8}"
EXTRA="${EXTRA:-}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
E=examples/MaterialsSaddles/experiments/hint_vs_noise/hint_vs_noise.py
if [ "$STAGE" = analyze ]; then
  "$PYTHON" -u "$E" analyze --out-dir "$OUT" $EXTRA
  exit 0
fi
GROUP="${GROUP:?Pass GROUP=0..3}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" -u "$E" rollout --out-dir "$OUT" --group "$GROUP" $EXTRA
for stage in sella verify; do
  for w in $(seq 0 $((WORKERS - 1))); do
    "$PYTHON" -u "$E" "$stage" --out-dir "$OUT" --group "$GROUP" --shard "$w" --nshards "$WORKERS" $EXTRA &
  done
  wait
done
