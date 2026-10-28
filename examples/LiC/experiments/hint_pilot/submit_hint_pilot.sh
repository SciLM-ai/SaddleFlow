#!/bin/bash
# Submit the whole LiC hint pilot: A0 after Sella and A3 on 4 GPUs each (array tasks), the mirror test on
# one GPU, then the analyses and the combined report once all three finish. Every job is --no-requeue.
#   bash examples/LiC/experiments/hint_pilot/submit_hint_pilot.sh [OUT]
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${1:-/scratch/gqs7zr/saddleflow/lic_hint_pilot}"
GPUS="${GPUS:-4}"
cd "$REPO"
S=examples/LiC/experiments/hint_pilot/run_hint_pilot.sh
[ -f "$OUT/a0/endpoints.traj" ] || PYTHONPATH="$REPO" "$PYTHON" examples/LiC/experiments/hint_pilot/hint_pilot.py export-a0 --out-dir "$OUT"
A0=$(sbatch --parsable --no-requeue --array=0-$((GPUS - 1)) --job-name=hp_a0 --export=ALL,MODE=a0-sella,OUT="$OUT" "$S")
MI=$(sbatch --parsable --no-requeue --job-name=hp_mirror --time=01:00:00 --export=ALL,MODE=mirror,OUT="$OUT" "$S")
A3=$(sbatch --parsable --no-requeue --array=0-$((GPUS - 1)) --job-name=hp_a3 --export=ALL,MODE=a3,OUT="$OUT" "$S")
AN=$(sbatch --parsable --no-requeue --job-name=hp_analyze --dependency=afterany:$A0:$MI:$A3 --partition=standard \
     --gres=none --constraint= --cpus-per-task=2 --mem=16G --time=01:00:00 --export=ALL,MODE=analyze,OUT="$OUT" "$S")
echo "A0 after Sella $A0, mirror $MI, A3 $A3, analysis $AN -> $OUT/hint_pilot_report.txt"
