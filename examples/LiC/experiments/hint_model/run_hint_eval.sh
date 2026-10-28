#!/bin/bash
#SBATCH --job-name=lic_hint_eval
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_hint_eval_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_hint_eval_%j.err
#SBATCH --no-requeue

# Evaluate A1, A2 and the untouched n12_uncond (B0) from the same hint starts: rollouts, Sella from every
# endpoint (WORKERS processes on the GPU), then the report.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
ROOT="${ROOT:-/scratch/gqs7zr/saddleflow/lic_hint_model}"
OUT="${OUT:-$ROOT/eval}"
WORKERS="${WORKERS:-4}"
B0=/scratch/gqs7zr/saddleflow/lic_source_study_seed42/n12_uncond/checkpoint_final
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
E=examples/LiC/experiments/hint_model/evaluate_hint_model.py
"$PYTHON" -u "$E" rollout --out-dir "$OUT" --arm "A1=$ROOT/input/checkpoint_final" \
  --arm "A2=$ROOT/startonly/checkpoint_final" --arm "B0=$B0"
for w in $(seq 0 $((WORKERS - 1))); do
  "$PYTHON" -u "$E" sella --out-dir "$OUT" --shard "$w" --nshards "$WORKERS" &
done
wait
"$PYTHON" -u "$E" analyze --out-dir "$OUT" --device cpu
