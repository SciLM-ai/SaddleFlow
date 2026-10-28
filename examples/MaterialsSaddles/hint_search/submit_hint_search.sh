#!/bin/bash
# Submit the mp20bat hint search: select (here, CPU) -> model rollouts (1 GPU) -> search (GPUS GPUs) ->
# verify (GPUS GPUs) -> analyze (CPU). Every job is --no-requeue. LIMIT=2 runs only the first two minima
# (the cost check); rerunning without LIMIT into the same OUT reuses every finished run.
#   LIMIT=2 bash examples/MaterialsSaddles/hint_search/submit_hint_search.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
RUN="${RUN:-/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_21421000}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/mp20bat_hint_search}"
GPUS="${GPUS:-4}"
LIMIT="${LIMIT:-0}"
cd "$REPO"
S=examples/MaterialsSaddles/hint_search/run_hint_search.sh
[ -f "$OUT/plan.json" ] || PYTHONPATH="$REPO" "$PYTHON" examples/MaterialsSaddles/hint_search/hint_search.py select --run-dir "$RUN" --out-dir "$OUT"
E="ALL,RUN=$RUN,OUT=$OUT,LIMIT=$LIMIT"
MO=$(sbatch --parsable --no-requeue --job-name=hs_model --time=04:00:00 --export="$E,MODE=model" "$S")
SE=$(sbatch --parsable --no-requeue --array=0-$((GPUS - 1)) --job-name=hs_search --dependency=afterok:$MO --export="$E,MODE=search" "$S")
VE=$(sbatch --parsable --no-requeue --array=0-$((GPUS - 1)) --job-name=hs_verify --dependency=afterany:$SE --export="$E,MODE=verify" "$S")
AN=$(sbatch --parsable --no-requeue --job-name=hs_analyze --dependency=afterany:$VE --partition=standard --gres=none \
     --constraint= --cpus-per-task=2 --mem=16G --time=01:00:00 --export="$E,MODE=analyze" "$S")
echo "model $MO, search $SE, verify $VE, analyze $AN -> $OUT/report.txt (LIMIT=$LIMIT)"
