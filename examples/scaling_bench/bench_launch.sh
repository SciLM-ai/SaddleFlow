#!/bin/bash
# =============================================================================
# bench_launch.sh — run ONE throughput-benchmark config inside an existing
# Slurm GPU allocation. Portable across clusters (Perlmutter A100, Vista GH200,
# future Grace-Blackwell). Times the REAL production training step (same model,
# same flags as examples/MP20Bat/run.sh) for a fixed number of optimizer steps
# and writes a single JSON via the --bench-output path in train.py.
#
# This is the inner step; drive it with sweep.sh, which loops GPU counts to
# build a scaling curve from one allocation.
#
# ---- Inputs (env vars; sensible defaults) -----------------------------------
#   NPROC               total GPUs (= processes) for THIS config        [REQUIRED]
#   BATCH               per-GPU batch size                              [default 16]
#   MAX_STEPS           total optimizer steps (warmup + timed window)   [default 230]
#   WARMUP              warmup steps before the timing window opens      [default 30]
#   LIMIT_TRIPLETS      dataset slice (kept tiny + identical per site)   [default 2048]
#   BENCH_OUT_DIR       where the JSON + shared scratch rundir live      [default $SCRATCH/saddleflow_bench]
#   BENCH_GPUS_PER_NODE GPUs per node (4 Perlmutter, 1 Vista GH200)      [default $SLURM_GPUS_ON_NODE or 4]
#   BENCH_MACHINE       label baked into the JSON filename               [default $NERSC_HOST or hostname -s]
#   BENCH_MODULES       space-separated Lmod modules to load             [default: cudatoolkit/12.9 if `module` exists]
#   SADDLEFLOW_PYTHON   python interpreter                              [default: Perlmutter conda env, else `python`]
#   SADDLEFLOW_REPO     repo root                                       [auto: two dirs up from this script]
#   FAIRCHEM_CACHE_DIR  cached UMA-S-1.2 checkpoint dir                  [default $SCRATCH/fairchem_cache]
#   NUM_WORKERS         dataloader workers (fix across sites for parity) [default 8]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SADDLEFLOW_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TRAIN_PY="$REPO/examples/MP20Bat/train.py"

: "${NPROC:?set NPROC (total GPUs for this config)}"
BATCH="${BATCH:-16}"
MAX_STEPS="${MAX_STEPS:-230}"
WARMUP="${WARMUP:-30}"
LIMIT_TRIPLETS="${LIMIT_TRIPLETS:-2048}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-${SCRATCH:?\$SCRATCH not set}/saddleflow_bench}"
GPN="${BENCH_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
MACHINE="${BENCH_MACHINE:-${NERSC_HOST:-$(hostname -s)}}"

# Default python: the Perlmutter conda env if present, else whatever's on PATH.
if [ -z "${SADDLEFLOW_PYTHON:-}" ]; then
    if [ -x /global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python ]; then
        SADDLEFLOW_PYTHON=/global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python
    else
        SADDLEFLOW_PYTHON=python
    fi
fi
PYTHON="$SADDLEFLOW_PYTHON"

# Module loads — site-specific. Default to Perlmutter's cudatoolkit; override
# with BENCH_MODULES="" to skip, or BENCH_MODULES="cuda/12.x foo" elsewhere.
if command -v module >/dev/null 2>&1; then
    for m in ${BENCH_MODULES-cudatoolkit/12.9}; do
        module load "$m" 2>/dev/null || true
    done
fi

export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$SCRATCH/fairchem_cache}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Map NPROC -> (nodes, tasks-per-node) given GPUs/node. For NPROC <= GPN it's a
# single-node subset; above that we use whole nodes (sweep only asks for counts
# that divide evenly: 1,2,4 then 8,16 at GPN=4).
if [ "$NPROC" -le "$GPN" ]; then
    NODES=1; TPN="$NPROC"
else
    NODES=$(( NPROC / GPN )); TPN="$GPN"
    if [ $(( NODES * GPN )) -ne "$NPROC" ]; then
        echo "[bench] NPROC=$NPROC is not 1..$GPN or a multiple of GPN=$GPN" >&2
        exit 2
    fi
fi

# Pin to the FIRST $NODES nodes of the allocation. This makes placement
# deterministic across configs and guarantees rank 0 lands on a known node.
# (Without --nodelist, `srun --nodes=1` may place rank 0 on ANY allocated node,
# not the allocation's first — so deriving MASTER_ADDR from the full-allocation
# nodelist head is wrong and the workers hang waiting for a store nobody hosts.)
# MASTER_ADDR is resolved INSIDE the step below, from the step's own nodelist.
mapfile -t _ALLOC_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}")
SUBSET_NODES=$(IFS=,; echo "${_ALLOC_NODES[*]:0:$NODES}")
MASTER_PORT=$(( 29500 + NPROC ))     # vary by config to dodge TIME_WAIT reuse
CVD=$(seq -s, 0 $(( GPN - 1 )))      # 0,1,2,3 on Perlmutter; 0 on a 1-GPU node

RUNDIR="$BENCH_OUT_DIR/_rundir"      # shared across configs → dataset stats computed once
OUT_JSON="$BENCH_OUT_DIR/${MACHINE}_n${NODES}_g${NPROC}_b${BATCH}.json"
mkdir -p "$BENCH_OUT_DIR" "$RUNDIR"

echo "============================================================"
echo "[bench] machine=$MACHINE  GPUs=$NPROC (${NODES}n × ${TPN}/n, ${GPN}/node)  batch=$BATCH (global $(( NPROC * BATCH )))"
echo "[bench] steps=$MAX_STEPS (warmup $WARMUP)  limit_triplets=$LIMIT_TRIPLETS  workers=$NUM_WORKERS"
echo "[bench] python=$PYTHON"
echo "[bench] nodes=$SUBSET_NODES  port=$MASTER_PORT  json=$OUT_JSON"
echo "============================================================"

# srun-native SPMD (one task per GPU). See examples/MaterialsSaddles/run.sh and
# CLAUDE.md "Multi-node launch" for why this beats `accelerate launch` here.
srun --nodelist="$SUBSET_NODES" --nodes="$NODES" --ntasks="$NPROC" \
     --ntasks-per-node="$TPN" --gpus-per-node="$GPN" --gpu-bind=none \
     --distribution=block \
  bash -c "
    set -euo pipefail
    # Rank 0 == SLURM_NODEID 0 == first host of THIS step's nodelist. Resolve it
    # at runtime so it's correct regardless of which physical nodes srun picked.
    export MASTER_ADDR=\$(scontrol show hostnames \"\${SLURM_STEP_NODELIST:-\${SLURM_NODELIST:-\$SLURM_JOB_NODELIST}}\" | head -n1)
    export MASTER_PORT=$MASTER_PORT
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    export CUDA_VISIBLE_DEVICES=$CVD
    export FAIRCHEM_CACHE_DIR=$FAIRCHEM_CACHE_DIR
    export PYTHONPATH=$PYTHONPATH
    if [ \"\$SLURM_LOCALID\" = \"0\" ] && [ \"\$SLURM_NODEID\" = \"0\" ]; then
      echo \"[bench] node0 \$(hostname) master=\$MASTER_ADDR rank=\$RANK world=\$WORLD_SIZE gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)\"
    fi
    exec $PYTHON $TRAIN_PY \
        --subset mp20bat \
        --output-dir $RUNDIR \
        --num-epochs 1000000 \
        --limit-triplets $LIMIT_TRIPLETS \
        --batch-size $BATCH \
        --learning-rate 1e-3 \
        --uma-lr 1e-4 \
        --warmup-steps 1000 \
        --ema-decay 0.9995 \
        --unfreeze-uma-all \
        --early-time-film-blocks 0,1,2,3 \
        --com-symmetric-loss \
        --xt-perturb-sigma 0.05 \
        --xt-target-correction \
        --xt-target-correction-t-floor 0.1 \
        --no-inject-force \
        --no-frozen-force-backbone \
        --no-endpoint-features \
        --no-dimer-residual \
        --eigenmode-aux-weight 0 \
        --num-workers $NUM_WORKERS \
        --log-every 50 \
        --save-every-epochs 1000000 \
        --val-every-epochs 1000000 \
        --max-steps $MAX_STEPS \
        --bench-warmup $WARMUP \
        --bench-output $OUT_JSON
  "

echo "[bench] wrote $OUT_JSON"
