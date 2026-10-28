#!/bin/bash
# Evaluate one fine-tune checkpoint, each stage its own job:
#   land (4 GPUs, one per group) -> report (CPU)                        stage 1, the landing points alone
#   -> sella (4 GPUs) -> verify (4 GPUs) -> report (CPU)                stage 2, unless STAGE1_ONLY=1
# A stage starts only if the previous one succeeded.
#   CKPT=<run>/checkpoint_step_00005000 bash submit_eval.sh
# Optional: AFTER=<job id> waits for that job (e.g. the fine-tune writing CKPT); EXTRA="--arms a,b,..." trims the
# arms (variables reach the jobs through the environment, so commas in EXTRA are safe).
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
CKPT="${CKPT:?Pass CKPT=<checkpoint dir>}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/hint_finetune/eval/$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
S="$REPO/examples/MaterialsSaddles/experiments/hint_finetune/run_eval.sh"
mkdir -p "$OUT"
cpu=(--partition=standard --gres= --constraint= --cpus-per-task=4 --mem=64G)
export OUT CKPT EXTRA="${EXTRA:-}"
after=""; [ -n "${AFTER:-}" ] && after="--dependency=afterok:$AFTER"
# Landing runs UMA-M in strict fp32: ~0.85 s per start on a B200 or RTX Pro 6000 (about 2.5 h per group),
# ~1.65 s on an A100 80GB (about 5 h), ~2.5 s on an A40 (too slow), so not the A40s or A6000s.
j=$(STAGE=land sbatch --parsable --no-requeue --job-name=eval_land $after --constraint="b200|rtxpro6000|a100_80gb" --array=0-3 --time=12:00:00 --export=ALL "$S"); echo "land $j"
r=$(STAGE=report sbatch --parsable --no-requeue --job-name=eval_report1 --dependency=afterok:$j "${cpu[@]}" --time=01:00:00 --export=ALL "$S"); echo "report (stage 1) $r"
if [ "${STAGE1_ONLY:-0}" != 1 ]; then
  j=$(STAGE=sella sbatch --parsable --no-requeue --job-name=eval_sella --dependency=afterok:$j --array=0-3 --time=12:00:00 --export=ALL "$S"); echo "sella $j"
  j=$(STAGE=verify sbatch --parsable --no-requeue --job-name=eval_verify --dependency=afterok:$j --array=0-3 --time=12:00:00 --export=ALL "$S"); echo "verify $j"
  j=$(STAGE=report sbatch --parsable --no-requeue --job-name=eval_report --dependency=afterok:$j "${cpu[@]}" --time=01:00:00 --export=ALL "$S"); echo "report $j"
fi
echo "output: $OUT"
