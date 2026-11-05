#!/bin/bash
# train_fm1.sh -- FM1's training run with this repository's code: UMA-M-1.1 trained for one epoch on all four MaterialsSaddles
# subsets with path starts (the model of the MaterialsSaddles paper). Run it on the first node of a Slurm allocation; it
# starts one rank per GPU with srun (SPMD, static rendezvous). FM1 used 64 GH200 nodes with one GPU each and 8 records per
# GPU (global batch 512); any GPU count that divides 512 gives the same schedule.
#
# usage:  OUT=<output dir> [PY=python] [GPUS_PER_NODE=1] [RESUME_FROM=<OUT>/checkpoint_step_NNNNN] bash examples/FM1/train_fm1.sh
# Data: the MaterialsSaddles v2 release under $MATERIALSSADDLES_ROOT (examples/MaterialsSaddles/data_prep.py). UMA-M-1.1 is
# fetched by fairchem (gated on Hugging Face: accept facebook/UMA's terms and log in).
set -euo pipefail
: "${OUT:?set OUT to the output directory}" "${MATERIALSSADDLES_ROOT:?set MATERIALSSADDLES_ROOT to the v2 data}"
PY=${PY:-python}; GPUS_PER_NODE=${GPUS_PER_NODE:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,); NN=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD=$(( NN * GPUS_PER_NODE ))
(( 512 % WORLD == 0 )) || { echo "the GPU count ($WORLD) must divide the global batch 512"; exit 1; }
BS=$(( 512 / WORLD ))
mkdir -p "$OUT"
START=""; [ -n "${RESUME_FROM:-}" ] && START="--resume-from $RESUME_FROM"   # resumes mid-epoch, skipping trained batches
# The settings FM1 ran with (provenance/scripts/launch_foundation.sh), with the per-GPU batch and warmup scaled to the GPU count.
ARGS="--all-subsets --output-dir $OUT --subset-repeat oc22=8,mp20bat=8,oc20=2 --task-name-map oc22=oc20 \
  --backbone uma-m-1p1 --mixed-precision no --grad-clip-norm 1.0 --batch-size $BS --num-epochs 1 \
  --learning-rate 1e-3 --uma-lr 3e-3 --warmup-steps $(( 250 * WORLD )) --ema-decay 0.9996 \
  --loss-type huber --huber-delta 0.05 \
  --delta-endpoint-channels 0 --path-start-prob 1.0 --path-noise-sigma 0.3 --path-u-power 2.0 \
  --log-every 50 --save-every-epochs 1 --save-every-steps 2000 --val-every-steps 2000 --max-val-records 4000 \
  --num-workers 2 --unfreeze-uma-all --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --head-depth 3 --attn-layers 0 \
  --com-symmetric-loss --xt-target-correction --xt-perturb-sigma 0.05 \
  --no-inject-force --no-dimer-residual --no-frozen-force-backbone --no-endpoint-features --eigenmode-aux-weight 0 $START"
echo "[fm1] $NN nodes x $GPUS_PER_NODE GPUs, $BS records per GPU -> $OUT  $(date)"
srun --nodelist="$NODES" --nodes=$NN --ntasks=$WORLD --ntasks-per-node=$GPUS_PER_NODE --unbuffered bash -c "
  export PYTHONPATH=$REPO PYTHONUNBUFFERED=1 MATERIALSSADDLES_ROOT=$MATERIALSSADDLES_ROOT
  export MASTER_ADDR=${NODES%%,*} MASTER_PORT=${MASTER_PORT:-29521} RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
  exec $PY -u $REPO/examples/MaterialsSaddles/train.py $ARGS" > "$OUT/train.log" 2>&1
echo "[fm1] done $(date)"
