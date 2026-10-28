#!/bin/bash
# The whole Sella determinism experiment: starts (here), tier 1 on an A100 80GB and on CPU, tier 2 conditions
# C0..C5 (one A100 80GB each) and C6..C8 (single-threaded CPU), then the report. Every job is --no-requeue.
#   OUT=<dir> EXTRA="<flags for tests>" bash examples/MaterialsSaddles/experiments/sella_determinism/submit_determinism.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
export OUT="${OUT:-/scratch/gqs7zr/saddleflow/sella_determinism}"
export EXTRA="${EXTRA:-}"
cd "$REPO"
D=examples/MaterialsSaddles/experiments/sella_determinism
S=$D/run_determinism.sh
CPU=(--partition=standard --gres=none --constraint= --cpus-per-task=16 --mem=64G)
[ -f "$OUT/starts.json" ] || PYTHONPATH="$REPO" "$PYTHON" $D/sella_determinism.py select --out-dir "$OUT"
T1=$(sbatch --parsable --no-requeue --job-name=det_forces --time=02:00:00 --export=ALL,STAGE=forces "$S")
T1C=$(sbatch --parsable --no-requeue --job-name=det_forces_cpu --time=03:00:00 "${CPU[@]}" --export=ALL,STAGE=forces_cpu,OMP_NUM_THREADS=16 "$S")
T2=$(sbatch --parsable --no-requeue --job-name=det_sella --array=0-5 --export=ALL,STAGE=sella "$S")
CPUJOBS=""
for C in C6 C7 C8; do
  J=$(sbatch --parsable --no-requeue --job-name=det_$C --time=12:00:00 "${CPU[@]}" --export=ALL,STAGE=cpu,CONDITION=$C,WORKERS=6 "$S")
  CPUJOBS="$CPUJOBS:$J"
done
AN=$(sbatch --parsable --no-requeue --job-name=det_analyze --time=01:00:00 --dependency=afterany:$T1:$T1C:$T2$CPUJOBS \
     --partition=standard --gres=none --constraint= --cpus-per-task=2 --mem=16G --export=ALL,STAGE=analyze "$S")
echo "tier 1 $T1 (GPU) $T1C (CPU); tier 2 $T2 (C0..C5) CPU${CPUJOBS} (C6 C7 C8); analysis $AN -> $OUT/report.txt"
