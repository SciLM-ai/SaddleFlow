#!/bin/bash
#SBATCH -N 128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -o /pscratch/sd/i/ilgar/SaddleFlow_MaterialsSaddles/logs/slurm_%j.out
#SBATCH -e /pscratch/sd/i/ilgar/SaddleFlow_MaterialsSaddles/logs/slurm_%j.err
#SBATCH -A m1883_g
#SBATCH -J saddleflow_MaterialsSaddles

# Mode-1 product-conditional flow matching, full UMA-S-1.2 unfreeze, 4-block
# time-FiLM, hybrid PBC-correct convergent v_target with σ=0.05 Å perturb,
# CoM-symmetric loss. Trains on the FULL MaterialsSaddles dataset (lemat +
# oc20 + oc22 + mp20bat = 34.1M triplets, 61.4M records after R↔P doubling,
# ~640 GiB on disk). Training only — eval is launched separately, per-subset,
# after the run finishes (see Phase-1 epilog for the command).
#
# Prereq (one-time): saddleflow is pip-installed in the conda env so
# `import saddleflow` works from anywhere on the compute nodes.
#     pip install -e /global/cfs/cdirs/m1883/ilgar/codes/SaddleFlow
#
# Submission:
#     cd /global/cfs/cdirs/m1883/ilgar/codes/SaddleFlow/examples/MaterialsSaddles
#     sbatch run.sh
# (SLURM stages run.sh into /var/spool, so we use $SLURM_SUBMIT_DIR — preserved
#  by SLURM — to locate train.py and the other example scripts.)
#
# Smoke test (inside an existing allocation, from the examples dir):
#     SMOKE=1 bash run.sh                 # 1 epoch × 4 triplets
#
# Resume an interrupted training:
#     RESUME_FROM=/pscratch/sd/i/ilgar/SaddleFlow_MaterialsSaddles/runs/PREV/checkpoint_epoch_NN \
#         sbatch run.sh

set -euo pipefail

# Locate train.py robustly. Two cases to handle:
#   (a) `bash run.sh` from examples/MaterialsSaddles — BASH_SOURCE resolves
#       to the real file path; SLURM_SUBMIT_DIR may point elsewhere (the dir
#       where salloc was run).
#   (b) `sbatch run.sh` from examples/MaterialsSaddles — SLURM stages the
#       script to /var/spool/slurmd/jobN/, so BASH_SOURCE points there (no
#       train.py); SLURM_SUBMIT_DIR is the dir that contains train.py.
# Try both candidates, pick whichever actually has train.py next to it.
SCRIPT_DIR=""
for _candidate in \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
    "${SLURM_SUBMIT_DIR:-}"; do
    if [ -n "$_candidate" ] && [ -f "$_candidate/train.py" ]; then
        SCRIPT_DIR="$_candidate"
        break
    fi
done
if [ -z "$SCRIPT_DIR" ]; then
    echo "[run] FATAL: cannot locate train.py." >&2
    echo "[run]   Tried BASH_SOURCE path and \$SLURM_SUBMIT_DIR." >&2
    echo "[run]   cd to examples/MaterialsSaddles before running sbatch/bash." >&2
    exit 1
fi
RUN_ROOT="$SCRATCH/SaddleFlow_MaterialsSaddles"
OUT_DIR="$RUN_ROOT/runs/MaterialsSaddles_$(date +%Y%m%d_%H%M%S)"
PYTHON=/global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python

mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"

# Module loads (Lmod) — NERSC Perlmutter. CUDA libs are also bundled inside the
# torch wheel (cu128); cudatoolkit/12.9 here just makes nvcc / system libs match.
module load cudatoolkit/12.9 2>/dev/null || true

# Perlmutter A100 nodes have 4 GPUs per node — use them all.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Where fairchem finds the cached UMA-S-1.2 checkpoint. Perlmutter compute
# nodes DO have outbound internet, so a missing cache would still work — but
# pointing at the existing cache avoids 512 ranks redundantly re-downloading
# from HuggingFace (slow, and risks rate-limiting). Hardcoded here so the run
# does not depend on the submitting shell's .bashrc having exported it.
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-/pscratch/sd/i/ilgar/fairchem_cache}"

# Production: train on the FULL MaterialsSaddles dataset (--all-subsets,
# ~34M triplets, ~640 GiB on disk). Hyperparameters below assume 128 nodes ×
# 4 A100s = 512 GPUs and per-process batch 8 → global batch 4096.
#
# SMOKE=1 swaps in tiny epoch / triplet counts so a `bash run.sh` inside an
# allocation completes in ~2 min and exercises the full code path (single
# subset: mp20bat only).
if [ -n "${SMOKE:-}" ]; then
    SUBSETS_ARG="--subsets mp20bat"
    NUM_EPOCHS=${SMOKE_EPOCHS:-1}
    LIMIT_TRIPLETS=${SMOKE_TRIPLETS:-4}
    SAVE_EVERY_EPOCHS=1
    SAVE_EVERY_STEPS=0
    VAL_EVERY_STEPS=0
    BATCH_SIZE=4
    HEAD_LR=1e-3
    WARMUP_STEPS_INTENT=50
    EMA_DECAY=0.99
    echo "[run] SMOKE mode: NUM_EPOCHS=$NUM_EPOCHS LIMIT_TRIPLETS=$LIMIT_TRIPLETS"
else
    SUBSETS_ARG="--all-subsets"
    # Dataset: 34.1M triplets, 61.4M records after R↔P doubling.
    # Default config: -N 128 × 4 GPUs × bs=8 → global batch 4096.
    # Steps/epoch = 61.44M / 4096 = 15k. 2 epochs × 15k = 30k steps.
    # Empirical per-step at 512 GPUs bs=8: ~5.6 s (measured job 53119627,
    # 2000 steps in 11.3k s). 30k × 5.6 ≈ 47 h training + ~4 h val overhead
    # (10 vals × ~25 min each, see VAL_EVERY_STEPS below) ≈ 51 h — needs the
    # 48 h walltime ceiling. If resuming from an existing checkpoint the
    # remaining-work budget is correspondingly smaller.
    NUM_EPOCHS=2
    LIMIT_TRIPLETS=0
    SAVE_EVERY_EPOCHS=1
    SAVE_EVERY_STEPS=2000              # ~3 h between saves at production speed
    VAL_EVERY_STEPS=3000               # 10 step-vals over 30k optimizer steps
    BATCH_SIZE=8
    # Intent: warmup over ~5 % of 30k optimizer steps = 1500 steps. The actual
    # value passed to train.py is multiplied by NUM_PROCS below to compensate
    # for accelerate's AcceleratedScheduler advancing the underlying scheduler
    # by num_processes ticks per .step() call (see accelerate/scheduler.py:75).
    # Without that compensation, --warmup-steps 1500 would complete in only
    # 1500 / NUM_PROCS ≈ 3 optimizer steps — that bug was confirmed in
    # job 53119627 (LR hit peak by step ~5).
    WARMUP_STEPS_INTENT=1500
    # EMA half-life ≈ ln 2/(1−decay) = 4.6k steps ≈ 15 % of 30k → Karras band.
    # If you change NUM_EPOCHS/BATCH_SIZE: decay = 1 - ln2/(0.10–0.20 × total_steps).
    EMA_DECAY=0.99985
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
# Derive port from job id to avoid collisions with other jobs sharing the head node.
MASTER_PORT=$((30000 + ${SLURM_JOB_ID:-0} % 30000))
NUM_NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=4
NUM_PROCS=$((NUM_NODES * GPUS_PER_NODE))
GLOBAL_BATCH=$((NUM_PROCS * BATCH_SIZE))

# AcceleratedScheduler advances the underlying LR scheduler by NUM_PROCS ticks
# per .step() call. Compensate so `--warmup-steps` is interpreted in
# OPTIMIZER-STEP units (the natural unit), not scheduler-tick units.
WARMUP_STEPS=$((WARMUP_STEPS_INTENT * NUM_PROCS))

# Head-LR sqrt-scaled from the mp20bat baseline (1e-3 @ global batch 192). We
# prefer sqrt over linear because (a) UMA is pretrained and large LR risks
# destroying its features, (b) sqrt is empirically stable for this kind of
# discriminative-LR finetune. UMA-blocks LR stays fixed at 1e-4 regardless of
# batch size — it's the pretrained anchor.
if [ -z "${HEAD_LR:-}" ]; then
    HEAD_LR=$($PYTHON -c "import math; print(f'{1e-3 * math.sqrt($GLOBAL_BATCH / 192):.2e}')")
fi

echo "============================================================"
echo "[run] Variant:    MaterialsSaddles (full dataset)"
echo "[run] Date:       $(date)"
echo "[run] Job ID:     ${SLURM_JOB_ID:-?}"
echo "[run] Nodes:      $NUM_NODES  (${SLURM_JOB_NODELIST:-local})"
echo "[run] GPUs total: $NUM_PROCS"
echo "[run] Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "[run] Scripts:    $SCRIPT_DIR"
echo "[run] Output:     $OUT_DIR"
echo "[run] Subsets:    $SUBSETS_ARG"
echo "[run] Per-GPU bs: $BATCH_SIZE  →  global batch: $GLOBAL_BATCH"
echo "[run] Head LR:    $HEAD_LR  (sqrt-scaled from 1e-3 @ batch 192)"
echo "[run] UMA LR:     1e-4  (fixed)"
echo "[run] Warmup:     $WARMUP_STEPS_INTENT optimizer steps  (passed as $WARMUP_STEPS to compensate AcceleratedScheduler ×$NUM_PROCS)    EMA decay: $EMA_DECAY"
echo "[run] Save:       every $SAVE_EVERY_EPOCHS epochs + every $SAVE_EVERY_STEPS steps"
echo "============================================================"

# ============================================================
# Phase 0 — Idempotent setup: precompute per-subset stats files
# ============================================================
# `train.py` looks up dataset_stats_<subset>.json at the canonical scratch
# location ($SCRATCH/MaterialsSaddles/). Computing them on the multi-rank hot
# path causes a fatal collective desync, so compute them ONCE here, single-
# process, before any srun. Idempotent — re-runs only fill in missing files.
MS_ROOT=$SCRATCH/MaterialsSaddles
NEED_PRECOMPUTE=0
for s in lemat oc20 oc22 mp20bat; do
    if [ ! -f "$MS_ROOT/dataset_stats_${s}.json" ]; then
        NEED_PRECOMPUTE=1
        break
    fi
done
if [ "$NEED_PRECOMPUTE" = "1" ]; then
    echo "[run] Precomputing per-subset stats (one-time setup) ..."
    $PYTHON -c "
import os
from saddleflow.data import MaterialsSaddlesDataset
root = os.path.expandvars('\$SCRATCH/MaterialsSaddles')
for s in ('lemat', 'oc20', 'oc22', 'mp20bat'):
    out = f'{root}/dataset_stats_{s}.json'
    if os.path.isfile(out):
        continue
    ds = MaterialsSaddlesDataset(f'{root}/{s}')
    ds.compute_stats(stats_cache=out, sample=512)
    print(f'  {s}: ⟨‖Δ‖⟩={ds.delta_norm_mean:.3f} Å → {out}')
"
else
    echo "[run] Per-subset stats already cached at $MS_ROOT/dataset_stats_*.json — skipping."
fi

# ------------------------------------------------------------
# Phase 0b — Pre-warm the UMA checkpoint cache (single process).
# ------------------------------------------------------------
# load_uma_backbone() -> get_predict_unit() has NO cross-rank serialization,
# unlike the dataset staging above. On a cold FAIRCHEM_CACHE_DIR all 512 ranks
# would race to download the same checkpoint into the same path simultaneously,
# which can interleave/partial-write and corrupt it. So fetch it ONCE here,
# single-process, before the srun fan-out. Warm cache → fast no-op load on CPU.
echo "[run] Pre-warming UMA checkpoint cache (single process) ..."
$PYTHON -c "
from saddleflow.utils import load_uma_backbone
load_uma_backbone('uma-s-1p2', device='cpu')
print('  UMA-S-1.2 checkpoint present in FAIRCHEM_CACHE_DIR')
"

# ============================================================
# Phase 1 — Training (uses ALL allocated nodes × 4 GPUs each)
# ============================================================
# 30-min NCCL watchdog so DDP init + UMA cache load + first allreduce don't blow
# past the default 10-min timeout on a slow Lustre day.
export NCCL_TIMEOUT=1800
#
# LAUNCH MODEL: srun-native SPMD (one task per GPU), NOT `accelerate launch`.
# -----------------------------------------------------------------------------
# We previously used `accelerate launch --num_machines N --rdzv_backend c10d`,
# which spins up torch-elastic's DYNAMIC cross-node rendezvous (128 agents
# negotiating quorum). At 128 nodes that reliably stranded ~2 agents and timed
# out (jobs 53338690 at 600s, 53401004 even at 1800s) — the failure was in the
# rendezvous protocol, not slowness, so widening the timeout did not help.
#
# fairchem/UMA itself does NOT use a cross-node c10d rendezvous for multinode
# (that path is single-node only, see slurm_launch.py local_launch). Its
# multinode path (common/distutils.py::setup) is SPMD: srun launches one task
# per GPU, and each rank derives RANK=SLURM_PROCID, LOCAL_RANK=SLURM_LOCALID,
# WORLD_SIZE=SLURM_NTASKS, then calls a STATIC init_process_group on
# tcp://<node0>:<port> — no quorum dance. We mirror that here.
#
# accelerate needs no launcher for this: when LOCAL_RANK is set it enters
# MULTI_GPU mode and init_process_group()s from env:// (state.py:202,244). So we
# just export the 5 standard torch-dist env vars from SLURM and run train.py
# directly. --gpu-bind=none so every task sees all 4 GPUs and accelerate's
# set_device(LOCAL_RANK) lands each rank on its own GPU (fairchem's model).
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
    if [ \"\$SLURM_LOCALID\" = \"0\" ]; then
      echo \"[node \$SLURM_NODEID] \$(hostname) localid0 rank=\$RANK world=\$WORLD_SIZE gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\\n' ',')\"
    fi
    $PYTHON $SCRIPT_DIR/train.py \
        $SUBSETS_ARG \
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
        --xt-perturb-sigma 0.05 \
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

echo "[run] Training finished at $(date). Checkpoint: $OUT_DIR/checkpoint_final"
echo ""
echo "[run] No automatic eval — run eval_full_testset_K10.py per-subset"
echo "[run] afterwards (it shards the test set across GPUs via accelerate,"
echo "[run] so just point --num_processes at all GPUs in your allocation):"
echo "[run]"
echo "[run]   for s in mp20bat oc22 oc20 lemat; do"
echo "[run]     accelerate launch --num_processes $NUM_PROCS --multi_gpu \\"
echo "[run]       --mixed_precision bf16 \\"
echo "[run]       $SCRIPT_DIR/eval_full_testset_K10.py \\"
echo "[run]       --ckpt-dir $OUT_DIR/checkpoint_final --subset \$s --K 10"
echo "[run]   done"
echo "[run]"
echo "[run] Use --num-cases N to subsample (lemat has 1.57 M test triplets;"
echo "[run] full K=10 sweep on lemat is multi-day even sharded across 64 GPUs)."
