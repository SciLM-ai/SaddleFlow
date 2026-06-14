#!/bin/bash
# =============================================================================
# sweep.sh — full scaling sweep from ONE allocation. Calls bench_launch.sh once
# per GPU count, all writing JSONs into BENCH_OUT_DIR for analyze.py to combine.
#
# Run it INSIDE an allocation that permits as many tasks/node as you have GPUs
# (this is the gotcha from CLAUDE.md: an salloc defaulting to 1 task/node cannot
# srun 4 tasks/node). On Perlmutter:
#
#   salloc -N 4 -C gpu -q interactive -t 01:00:00 -A m1883_g \
#          --ntasks-per-node=4 --gpus-per-node=4 \
#          bash examples/scaling_bench/sweep.sh
#
# On Vista (GH200, 1 GPU/node): see examples/scaling_bench/README.md — set
# BENCH_GPUS_PER_NODE=1, BENCH_MACHINE=vista, GPU_LIST="1 2 4 8", and your own
# python/modules/account.
#
# ---- Inputs (env vars) ------------------------------------------------------
#   GPU_LIST        space-separated GPU counts to benchmark  [default "1 2 4 8 16"]
#   BATCH           per-GPU batch for the scaling curve       [default 16]
#   BATCH_LIST      if set, ALSO sweep these per-GPU batches on 1 GPU
#                   (memory-headroom probe; e.g. "16 32 48 64")
#   BENCH_OUT_DIR   output dir                                [default $SCRATCH/saddleflow_bench]
#   ... plus everything bench_launch.sh reads (BENCH_GPUS_PER_NODE, BENCH_MACHINE,
#       SADDLEFLOW_PYTHON, BENCH_MODULES, MAX_STEPS, WARMUP, LIMIT_TRIPLETS, ...).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-${SCRATCH:?\$SCRATCH not set}/saddleflow_bench}"
GPU_LIST="${GPU_LIST:-1 2 4 8 16}"
BATCH="${BATCH:-16}"
GPN="${BENCH_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

# Cap GPU_LIST to what the allocation actually has.
ALLOC_GPUS=$(( ${SLURM_NNODES:-1} * GPN ))
echo "[sweep] allocation: ${SLURM_NNODES:-1} node(s) × ${GPN} GPU/node = ${ALLOC_GPUS} GPUs"
echo "[sweep] requested GPU_LIST: $GPU_LIST  (batch $BATCH)  out=$BENCH_OUT_DIR"

# Phase A — weak scaling at fixed per-GPU batch.
for n in $GPU_LIST; do
    if [ "$n" -gt "$ALLOC_GPUS" ]; then
        echo "[sweep] skip n=$n (> $ALLOC_GPUS allocated)"
        continue
    fi
    echo "[sweep] --- scaling config: $n GPU(s) ---"
    NPROC="$n" BATCH="$BATCH" bash "$SCRIPT_DIR/bench_launch.sh"
done

# Phase B (optional) — single-GPU batch-size / memory-headroom sweep.
if [ -n "${BATCH_LIST:-}" ]; then
    echo "[sweep] === single-GPU batch-headroom sweep: $BATCH_LIST ==="
    for b in $BATCH_LIST; do
        echo "[sweep] --- batch config: 1 GPU × bs=$b ---"
        # `|| true` so an OOM at a big batch doesn't abort the whole sweep.
        NPROC=1 BATCH="$b" bash "$SCRIPT_DIR/bench_launch.sh" || \
            echo "[sweep] bs=$b failed (likely OOM) — recorded as the ceiling"
    done
fi

echo "[sweep] done. JSONs in $BENCH_OUT_DIR"
echo "[sweep] analyze with:  python $SCRIPT_DIR/analyze.py --results-dir $BENCH_OUT_DIR --out-dir $BENCH_OUT_DIR/report"
