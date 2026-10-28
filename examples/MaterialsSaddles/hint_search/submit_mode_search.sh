#!/bin/bash
# Stage 2 (mode search) then stage 1 (new verification of every converged point, old and new) then the
# comparison report, on the two cost-check minima. GPUS GPUs per stage; every job is --no-requeue.
set -euo pipefail
cd "${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
S=examples/MaterialsSaddles/hint_search/run_mode_search.sh
GPUS="${GPUS:-2}"
SE=$(sbatch --parsable --no-requeue --array=0-$((GPUS - 1)) --job-name=ms_search --export=ALL,MODE=search "$S")
VE=$(sbatch --parsable --no-requeue --array=0-$((GPUS - 1)) --job-name=ms_verify --dependency=afterany:$SE --export=ALL,MODE=verify "$S")
AN=$(sbatch --parsable --no-requeue --job-name=ms_analyze --dependency=afterany:$VE --partition=standard --gres=none \
     --constraint= --cpus-per-task=2 --mem=16G --time=01:00:00 --export=ALL,MODE=analyze "$S")
echo "search $SE, verify $VE, analyze $AN -> /scratch/gqs7zr/saddleflow/mp20bat_hint_search/mode_report.txt"
