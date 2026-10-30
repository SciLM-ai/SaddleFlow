#!/bin/bash
#SBATCH --job-name=steer_test
#SBATCH --partition=gpu
#SBATCH --constraint=h100_94gb|a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/bigtemp/gqs7zr/saddleflow/steer_test_%x_%j.out
#SBATCH --error=/bigtemp/gqs7zr/saddleflow/steer_test_%x_%j.err
#SBATCH --no-requeue

# steer_test.py on the CS cluster, one GPU (data and caches under /bigtemp, weights offline, as run_eval_cs.sh).
#   STAGE=refine   the stored saddles on UMA-S, WORKERS processes on the GPU
#   STAGE=test     land (one process), then Sella (WORKERS processes), then the report
# Every stage skips work already written: rerun STAGE=test after adding a missing checkpoint.
#   export TMPDIR=/tmp
#   r=$(STAGE=refine sbatch --parsable --job-name=steer_refine --export=ALL run_steer_test_cs.sh)
#   STAGE=test sbatch --job-name=steer_test --dependency=afterok:$r --export=ALL run_steer_test_cs.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/u/gqs7zr/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/u/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/bigtemp/gqs7zr/saddleflow}"
OUT="${OUT:-$SADDLEFLOW_DATA/steer_test}"
STAGE="${STAGE:?Pass STAGE=refine|test}"
WORKERS="${WORKERS:-8}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CACHE=/bigtemp/gqs7zr/cache
export HF_HOME="${HF_HOME:-$CACHE/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE/inductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE/xdg}" FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$CACHE/fairchem}"
export OMP_NUM_THREADS=2
E=examples/MaterialsSaddles/experiments/steer_test/steer_test.py
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
nvidia-smi --query-gpu=name --format=csv,noheader
# WORKERS shards of one stage on the GPU; fail the job if any shard failed (a bare `wait` returns 0 regardless).
sharded() {
  local pids=() status=0
  for w in $(seq 0 $((WORKERS - 1))); do
    "$PYTHON" -u "$E" "$1" --out-dir "$OUT" --shard "$w" --nshards "$WORKERS" & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || { echo "a $1 worker failed; see the .err file" >&2; exit 1; }
}
case "$STAGE" in
  refine)
    sharded refine
    "$PYTHON" -u "$E" report --out-dir "$OUT" ;;
  test)
    "$PYTHON" -u "$E" land --out-dir "$OUT"
    sharded sella
    "$PYTHON" -u "$E" report --out-dir "$OUT" ;;
esac
