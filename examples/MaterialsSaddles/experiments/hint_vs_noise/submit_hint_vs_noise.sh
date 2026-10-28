#!/bin/bash
# The 30-minimum comparison: plan.json (if missing), one GPU job per group, then the report.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/hint_vs_noise}"
D="$REPO/examples/MaterialsSaddles/experiments/hint_vs_noise"
mkdir -p "$OUT"
[ -f "$OUT/plan.json" ] || PYTHONPATH="$REPO" "$PYTHON" "$D/hint_vs_noise.py" select --out-dir "$OUT"
ids=()
for g in 0 1 2 3; do
  ids+=("$(sbatch --parsable --no-requeue --job-name="hvn_g$g" --export=ALL,OUT="$OUT",GROUP="$g" "$D/run_hint_vs_noise.sh")")
done
dep=$(IFS=:; echo "${ids[*]}")
sbatch --no-requeue --job-name=hvn_analyze --partition=standard --gres= --constraint= --cpus-per-task=4 --mem=32G \
  --time=02:00:00 --dependency="afterany:$dep" --export=ALL,OUT="$OUT",STAGE=analyze "$D/run_hint_vs_noise.sh"
echo "groups: ${ids[*]}"
