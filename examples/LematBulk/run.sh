#!/bin/bash
#SBATCH -N 128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -o /pscratch/sd/i/ilgar/SaddleFlow_LematBulk/logs/slurm_%j.out
#SBATCH -e /pscratch/sd/i/ilgar/SaddleFlow_LematBulk/logs/slurm_%j.err
#SBATCH -A m1883_g
#SBATCH -J saddleflow_lematbulk
#
# lemat-bulk DATASET 1 (both-converged R/S/P triplets) — Mode-1
# product-conditional flow matching, x0 = (R+P)/2 midpoint. This is the EXACT
# production recipe from examples/MaterialsSaddles/run.sh (full UMA unfreeze,
# 4-block time-FiLM, convergent v_target t_floor=0.1, CoM-symmetric loss),
# trained on the local lemat-bulk D1 shards (no HuggingFace staging).
#
#   MODEL 1 (reference):       sbatch run.sh                  # SIGMA=0.05
#   MODEL 2 (wider off-line):  sbatch --export=ALL,SIGMA=0.15,TAG=model2 run.sh
#
# Sizing: global batch 4096 (128 nodes x 4 x bs8). D1 train = 1,677,551 triplets
# x2 (R<->P doubling) = 3,355,102 records -> 819 steps/epoch. NUM_EPOCHS=18 ->
# ~14.7k optimizer steps (~23 h at ~5.6 s/step, fits the 24 h wall). EMA decay
# 0.99969 (15%-of-run half-life at ~15k steps). LR unchanged from production
# (keys off the global batch, not the dataset size). Intra-epoch checkpoints +
# RESUME_FROM cover a wall-clock overrun.
#
# Smoke test inside an existing allocation:  SMOKE=1 bash run.sh

set -euo pipefail

# This script lives in examples/LematBulk/; train.py + data_prep.py live in the
# sibling examples/MaterialsSaddles/. Resolve both robustly under sbatch (which
# stages the script into /var/spool) via $SLURM_SUBMIT_DIR fallback.
THIS_DIR=""
for _c in \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
    "${SLURM_SUBMIT_DIR:-}"; do
    if [ -n "$_c" ] && [ -f "$_c/run.sh" ] && [ -d "$_c/../MaterialsSaddles" ]; then
        THIS_DIR="$_c"; break
    fi
done
if [ -z "$THIS_DIR" ]; then
    echo "[run] FATAL: cannot locate examples/LematBulk/run.sh — cd there before sbatch." >&2
    exit 1
fi
TRAIN_DIR="$(cd "$THIS_DIR/../MaterialsSaddles" && pwd)"   # train.py + data_prep.py

# --- local lemat-bulk DATASET 1 ---
DATA_ROOT=/pscratch/sd/i/ilgar/genTS/lematbulk_Elnara_ls6redo
D1_SHARDS=$DATA_ROOT/doubleopt/aselmdb_no-EF
D1_MANIFEST=$DATA_ROOT/datasets/dataset1_split_manifest.csv
SUBSET=lematbulk_d1

# --- per-model knobs (env-overridable) ---
SIGMA=${SIGMA:-0.05}            # 0.05 = model 1 (reference); 0.15 = model 2
TAG=${TAG:-model1}

PYTHON=/global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python
RUN_ROOT=$SCRATCH/SaddleFlow_LematBulk
OUT_DIR=$RUN_ROOT/runs/${TAG}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"

module load cudatoolkit/12.9 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-/pscratch/sd/i/ilgar/fairchem_cache}"

if [ -n "${SMOKE:-}" ]; then
    NUM_EPOCHS=${SMOKE_EPOCHS:-1}
    LIMIT_TRIPLETS=${SMOKE_TRIPLETS:-8}
    SAVE_EVERY_EPOCHS=1; SAVE_EVERY_STEPS=0; VAL_EVERY_STEPS=0
    BATCH_SIZE=4; HEAD_LR=1e-3; WARMUP_STEPS_INTENT=10; EMA_DECAY=0.99
    echo "[run] SMOKE: epochs=$NUM_EPOCHS limit=$LIMIT_TRIPLETS sigma=$SIGMA"
else
    NUM_EPOCHS=${NUM_EPOCHS:-18}        # ~14.7k steps at global batch 4096
    LIMIT_TRIPLETS=0
    SAVE_EVERY_EPOCHS=1
    SAVE_EVERY_STEPS=1000               # ~1.6 h between saves at production speed
    VAL_EVERY_STEPS=1500               # ~10 step-vals over the run
    BATCH_SIZE=8
    WARMUP_STEPS_INTENT=750            # ~5% of ~15k steps
    EMA_DECAY=0.99969                  # 15%-of-run half-life at ~15k steps
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=$((30000 + ${SLURM_JOB_ID:-0} % 30000))
NUM_NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=4
NUM_PROCS=$((NUM_NODES * GPUS_PER_NODE))
GLOBAL_BATCH=$((NUM_PROCS * BATCH_SIZE))
# Compensate accelerate's AcceleratedScheduler (advances the LR scheduler by
# NUM_PROCS ticks per .step()) so --warmup-steps is in optimizer-step units.
WARMUP_STEPS=$((WARMUP_STEPS_INTENT * NUM_PROCS))
# Head LR sqrt-scaled from the mp20bat baseline (1e-3 @ global batch 192), same
# as production; UMA-blocks LR fixed at 1e-4.
if [ -z "${HEAD_LR:-}" ]; then
    HEAD_LR=$($PYTHON -c "import math; print(f'{1e-3 * math.sqrt($GLOBAL_BATCH / 192):.2e}')")
fi

echo "============================================================"
echo "[run] lemat-bulk D1   TAG=$TAG   sigma=$SIGMA"
echo "[run] Nodes: $NUM_NODES  GPUs: $NUM_PROCS  per-GPU bs: $BATCH_SIZE  global batch: $GLOBAL_BATCH"
echo "[run] Epochs: $NUM_EPOCHS   Head LR: $HEAD_LR   UMA LR: 1e-4   EMA: $EMA_DECAY"
echo "[run] Shards: $D1_SHARDS"
echo "[run] Split:  $D1_MANIFEST"
echo "[run] Output: $OUT_DIR"
echo "============================================================"

# Phase 0 — D1 delta_norm stats (informational; written where train.py looks).
MS_ROOT=$SCRATCH/MaterialsSaddles
mkdir -p "$MS_ROOT"
if [ ! -f "$MS_ROOT/dataset_stats_${SUBSET}.json" ]; then
    echo "[run] computing D1 delta_norm stats (one-time, single process) ..."
    $PYTHON -c "
from saddleflow.data import MaterialsSaddlesDataset
ds = MaterialsSaddlesDataset('$D1_SHARDS')
ds.compute_stats(stats_cache='$MS_ROOT/dataset_stats_${SUBSET}.json', sample=512)
print(f'  D1: <|Delta|>={ds.delta_norm_mean:.3f} A')
"
fi

# Phase 0b — pre-warm the UMA checkpoint cache (single process, avoids 512-rank race).
echo "[run] pre-warming UMA checkpoint cache ..."
$PYTHON -c "from saddleflow.utils import load_uma_backbone; load_uma_backbone('uma-s-1p2', device='cpu'); print('  UMA-S-1.2 cached')"

# Phase 1 — training (srun-native SPMD, one task per GPU; the canonical launcher).
export NCCL_TIMEOUT=1800
srun --ntasks=$NUM_PROCS --ntasks-per-node=$GPUS_PER_NODE \
     --gpus-per-node=$GPUS_PER_NODE --gpu-bind=none \
  bash -c "
    set -euo pipefail
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    $PYTHON $TRAIN_DIR/train.py \
        --subsets $SUBSET \
        --shards-dir $D1_SHARDS \
        --split-manifest $D1_MANIFEST \
        --output-dir $OUT_DIR \
        --num-epochs $NUM_EPOCHS \
        --limit-triplets $LIMIT_TRIPLETS \
        --batch-size $BATCH_SIZE \
        --learning-rate $HEAD_LR \
        --uma-lr 1e-4 \
        --warmup-steps $WARMUP_STEPS \
        --ema-decay $EMA_DECAY \
        --unfreeze-uma-all \
        --early-time-film-blocks 0,1,2,3 \
        --com-symmetric-loss \
        --xt-perturb-sigma $SIGMA \
        --xt-target-correction \
        --xt-target-correction-t-floor 0.1 \
        --no-inject-force \
        --no-frozen-force-backbone \
        --no-endpoint-features \
        --no-dimer-residual \
        --eigenmode-aux-weight 0 \
        --num-workers 2 \
        --log-every 50 \
        --save-every-epochs $SAVE_EVERY_EPOCHS \
        --save-every-steps $SAVE_EVERY_STEPS \
        --val-every-steps $VAL_EVERY_STEPS \
        ${RESUME_FROM:+--resume-from $RESUME_FROM}
  "

echo "[run] done $(date). Checkpoint: $OUT_DIR/checkpoint_final"
