#!/bin/bash
# The whole dimer determinism experiment on the Sella experiment's starts: conditions D0..D5 (one A100 80GB each)
# and D6..D9 (single-threaded CPU), then the report. Every job is --no-requeue.
#   OUT=<dir> EXTRA="<flags for tests>" bash examples/MaterialsSaddles/experiments/dimer_determinism/submit_determinism.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
export OUT="${OUT:-/scratch/gqs7zr/saddleflow/dimer_determinism}"
export EXTRA="${EXTRA:-}"
cd "$REPO"
S=examples/MaterialsSaddles/experiments/dimer_determinism/run_determinism.sh
CPU=(--partition=standard --gres=none --constraint= --cpus-per-task=16 --mem=64G)
T2=$(sbatch --parsable --no-requeue --job-name=dimer_det --array=0-5 --export=ALL,STAGE=dimer "$S")
CPUJOBS=""
for C in D6 D7 D8 D9; do
  J=$(sbatch --parsable --no-requeue --job-name=dimer_det_$C --time=12:00:00 "${CPU[@]}" --export=ALL,STAGE=cpu,CONDITION=$C,WORKERS=6 "$S")
  CPUJOBS="$CPUJOBS:$J"
done
AN=$(sbatch --parsable --no-requeue --job-name=dimer_det_analyze --time=01:00:00 --dependency=afterany:$T2$CPUJOBS \
     --partition=standard --gres=none --constraint= --cpus-per-task=2 --mem=16G --export=ALL,STAGE=analyze "$S")
echo "GPU $T2 (D0..D5) CPU${CPUJOBS} (D6..D9); analysis $AN -> $OUT/report.txt"
