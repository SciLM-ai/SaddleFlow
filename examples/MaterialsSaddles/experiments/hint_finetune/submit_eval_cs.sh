#!/bin/bash
# submit_eval.sh for the CS cluster (run_eval_cs.sh). Evaluate one fine-tune checkpoint, each stage its own job:
#   land (4 GPUs, one per group) -> report (CPU)                        stage 1, the landing points alone
#   -> sella (4 GPUs) -> verify (4 GPUs) -> report (CPU)                stage 2, unless STAGE1_ONLY=1
# A stage starts only if the previous one succeeded.
#   CKPT=<run>/checkpoint_final bash submit_eval_cs.sh
# Optional: AFTER=<job id> waits for that job (e.g. the fine-tune writing CKPT); EXTRA="--arms a,b,..." trims the
# arms (variables reach the jobs through the environment, so commas in EXTRA are safe).
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/u/gqs7zr/SaddleFlow}"
DATA="${SADDLEFLOW_DATA:-/bigtemp/gqs7zr/saddleflow}"
CKPT="${CKPT:?Pass CKPT=<checkpoint dir>}"
OUT="${OUT:-$DATA/hint_finetune/eval/$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
S="$REPO/examples/MaterialsSaddles/experiments/hint_finetune/run_eval_cs.sh"
# The jobs read the test minima from the failure-mode run that plan.json names (found under DATA on this cluster).
RUN="$DATA/$(basename "$(python3 -c "import json; print(json.load(open('$DATA/hint_vs_noise/plan.json'))['run_dir'])")")"
[ -s "$RUN/minima.traj" ] || { echo "no $RUN/minima.traj: copy the failure-mode run from Rivanna first" >&2; exit 1; }
[ -s "$CKPT/ema.pt" ] || [ -n "${AFTER:-}" ] || { echo "no $CKPT/ema.pt" >&2; exit 1; }
mkdir -p "$OUT"
cpu=(--partition=cpu --gres= --constraint= --cpus-per-task=4 --mem=64G)
# TMPDIR=/tmp: otherwise the submitting shell's TMPDIR reaches the jobs and slurmstepd may not be able to create it.
export OUT CKPT EXTRA="${EXTRA:-}" TMPDIR=/tmp
after=""; [ -n "${AFTER:-}" ] && after="--dependency=afterok:$AFTER"
# Landing runs UMA-M in strict fp32 (on Rivanna ~0.85 s per start on a B200 or RTX Pro 6000, about 2.5 h per
# group; ~1.65 s on an A100 80GB), so only the H100s and the A100 80GB; Sella and verification run UMA-S and take
# any of run_eval_cs.sh's GPUs.
j=$(STAGE=land sbatch --parsable --no-requeue --job-name=eval_land $after --constraint="h100_94gb|a100_80gb" --array=0-3 --time=12:00:00 --export=ALL "$S"); echo "land $j"
r=$(STAGE=report sbatch --parsable --no-requeue --job-name=eval_report1 --dependency=afterok:$j "${cpu[@]}" --time=01:00:00 --export=ALL "$S"); echo "report (stage 1) $r"
if [ "${STAGE1_ONLY:-0}" != 1 ]; then
  j=$(STAGE=sella sbatch --parsable --no-requeue --job-name=eval_sella --dependency=afterok:$j --array=0-3 --time=12:00:00 --export=ALL "$S"); echo "sella $j"
  j=$(STAGE=verify sbatch --parsable --no-requeue --job-name=eval_verify --dependency=afterok:$j --array=0-3 --time=12:00:00 --export=ALL "$S"); echo "verify $j"
  j=$(STAGE=report sbatch --parsable --no-requeue --job-name=eval_report --dependency=afterok:$j "${cpu[@]}" --time=01:00:00 --export=ALL "$S"); echo "report $j"
fi
echo "output: $OUT"
