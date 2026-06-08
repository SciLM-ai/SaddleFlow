#!/bin/bash
# eval_mp20bat_testset.sh — produce the paper test-set figures for ONE checkpoint.
#
# Writes, into OUT_DIR:
#   - parity_all_atoms_log.{png,pdf}        all-atom PBC-RMSD parity (SaddleFlow vs (R+P)/2)
#   - parity_maxdisp_log.{png,pdf}          max-atom-displacement parity
#   - hist_rmsd_K10_vs_{truth,baseline}.png
#   - hist_maxdisp_K10_vs_{truth,baseline}.png
#   - results.npz  cases.pkl  summary.json
#
# LAUNCH MODEL (important): single-node `accelerate launch` over the GPUs on the
# CURRENT compute node — NO srun, NO multi-node rendezvous. Run from inside an
# interactive GPU allocation (salloc) or a batch job. The 1737-case mp20bat test
# split takes ~5 min on 4 A100s; one node is plenty. Do NOT try to srun a
# multi-node step for this (see "gotchas" in the top-level CLAUDE.md).
#
# Usage:
#   bash eval_mp20bat_testset.sh <CKPT_DIR> <OUT_DIR> [live|ema] [subset] [K] [NUM_CASES]
#
#   NUM_CASES (6th arg, default 0): 0 = whole test split. N > 0 = a RANDOM
#   subset of N triplets (seeded, reproducible). Use this for the big subsets
#   (lemat/oc20/oc22) whose full test splits are 25k–4.7M cases — 2000 random
#   is a representative sample and runs in minutes. IMPORTANT: random, not
#   first-N — the parquet splits are sorted by ms_id, so a first-N slice is a
#   biased contiguous block. The wrapper passes `--random-sample` whenever
#   NUM_CASES > 0.
#
# Example 1 — full mp20bat test set, live weights (the paper figure):
#   bash eval_mp20bat_testset.sh \
#     $SCRATCH/SaddleFlow_MaterialsSaddles/runs/MaterialsSaddles_20260529_125142/checkpoint_final \
#     $SCRATCH/SaddleFlow_MaterialsSaddles/eval_mp20bat_test_K10_NEWMODEL/live \
#     live mp20bat 10
#
# Example 2 — 2000 random cases from each big subset (the cross-subset sweep):
#   CKPT=$SCRATCH/SaddleFlow_MaterialsSaddles/runs/MaterialsSaddles_20260529_125142/checkpoint_final
#   for s in lemat oc20 oc22; do
#     bash eval_mp20bat_testset.sh "$CKPT" \
#       $SCRATCH/SaddleFlow_MaterialsSaddles/eval_subsets_K10_NEWMODEL/$s \
#       live "$s" 10 2000
#   done
set -uo pipefail

CKPT="${1:?usage: eval_mp20bat_testset.sh CKPT_DIR OUT_DIR [live|ema] [subset] [K] [NUM_CASES]}"
OUT="${2:?need OUT_DIR}"
WEIGHTS="${3:-live}"     # live = model.safetensors  |  ema = ema.pt shadow (--use-ema)
SUBSET="${4:-mp20bat}"
K="${5:-10}"
NUM_CASES="${6:-0}"      # 0 = full test split; N>0 = N RANDOM seeded cases

# --- environment (load-bearing) -------------------------------------------------
# Conda env that has saddleflow + fairchem installed. Either `conda activate
# saddleflow` first, or point SADDLEFLOW_PYTHON at its python.
PYTHON="${SADDLEFLOW_PYTHON:-/global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python}"
# Cached UMA-S-1.2 checkpoint (matches train-time cache; avoids re-download).
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$SCRATCH/fairchem_cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true   # let accelerate see all node GPUs

EMA_FLAG=""
[ "$WEIGHTS" = "ema" ] && EMA_FLAG="--use-ema"

# NUM_CASES > 0 → random seeded subset (NOT first-N; see header).
SUBSET_FLAGS=""
[ "$NUM_CASES" -gt 0 ] 2>/dev/null && SUBSET_FLAGS="--num-cases $NUM_CASES --random-sample"

# GPUs on THIS node (1 accelerate process per GPU).
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
{ [ "$NPROC" -ge 1 ] 2>/dev/null; } || NPROC=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$OUT"

echo "[eval] ckpt=$CKPT"
echo "[eval] out=$OUT  weights=$WEIGHTS  subset=$SUBSET  K=$K  gpus=$NPROC  num_cases=${NUM_CASES} (0=full)"

# 1) Sample + score the test split → results.npz, summary.json, histograms.
"$PYTHON" -m accelerate.commands.launch \
    --num_processes "$NPROC" --multi_gpu --mixed_precision bf16 \
    eval_full_testset_K10.py \
    --ckpt-dir "$CKPT" --subset "$SUBSET" --K "$K" --seed 0 \
    --output-dir "$OUT" $EMA_FLAG $SUBSET_FLAGS

# 2) Paper-style parity plots (no re-sampling; reads results.npz).
"$PYTHON" replot_parity.py "$OUT"

echo "[eval] done — figures + summary.json in: $OUT"
