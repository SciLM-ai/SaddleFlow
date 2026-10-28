#!/bin/bash
# Tier B, each stage its own job: select (CPU) -> screen (1 GPU) -> discover (4 GPUs, one per group)
# -> verify (4 GPUs) -> labels (CPU) -> checks (CPU). A stage starts only if the previous one succeeded.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/hint_finetune/tierb}"
S="$REPO/examples/MaterialsSaddles/experiments/hint_finetune/run_tierb.sh"
mkdir -p "$OUT"
cpu=(--partition=standard --gres= --constraint= --cpus-per-task=4 --mem=64G)
dep=""
if [ ! -f "$OUT/candidates.traj" ]; then
  j=$(sbatch --parsable --no-requeue --job-name=tierb_select "${cpu[@]}" --time=02:00:00 --export=ALL,OUT="$OUT",STAGE=select "$S")
  dep="--dependency=afterok:$j"; echo "select $j"
fi
j=$(sbatch --parsable --no-requeue --job-name=tierb_screen $dep --time=03:00:00 --export=ALL,OUT="$OUT",STAGE=screen "$S"); echo "screen $j"
j=$(sbatch --parsable --no-requeue --job-name=tierb_discover --dependency=afterok:$j --array=0-3 --time=1-12:00:00 --export=ALL,OUT="$OUT",STAGE=discover "$S"); echo "discover $j"
j=$(sbatch --parsable --no-requeue --job-name=tierb_verify --dependency=afterok:$j --array=0-3 --time=1-00:00:00 --export=ALL,OUT="$OUT",STAGE=verify "$S"); echo "verify $j"
j=$(sbatch --parsable --no-requeue --job-name=tierb_labels --dependency=afterok:$j "${cpu[@]}" --time=02:00:00 --export=ALL,OUT="$OUT",STAGE=labels "$S"); echo "labels $j"
j=$(sbatch --parsable --no-requeue --job-name=tierb_checks --dependency=afterok:$j "${cpu[@]}" --time=02:00:00 --export=ALL,OUT="$OUT",STAGE=checks "$S"); echo "checks $j"
