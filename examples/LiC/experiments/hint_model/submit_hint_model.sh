#!/bin/bash
# Train A1 and A2 side by side (one GPU each), then evaluate both and the untouched n12_uncond.
#   bash examples/LiC/experiments/hint_model/submit_hint_model.sh
set -euo pipefail
cd "${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
D=examples/LiC/experiments/hint_model
A1=$(sbatch --parsable --no-requeue --job-name=hint_A1 --export=ALL,ARM=input $D/run_hint_train.sh)
A2=$(sbatch --parsable --no-requeue --job-name=hint_A2 --export=ALL,ARM=startonly $D/run_hint_train.sh)
EV=$(sbatch --parsable --no-requeue --dependency=afterok:$A1:$A2 $D/run_hint_eval.sh)
echo "A1 $A1, A2 $A2, evaluation $EV -> /scratch/gqs7zr/saddleflow/lic_hint_model/eval/report.txt"
