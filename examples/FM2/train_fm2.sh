#!/bin/bash
# train_fm2.sh -- FM2's training run with this repository's code: FM1's released weights fine-tuned for one epoch on all four
# MaterialsSaddles subsets with mover-set starts and a per-atom flag. Run it on the first node of a Slurm allocation; it
# starts one rank per GPU with srun (SPMD, static rendezvous, as examples/MaterialsSaddles/run.sh). FM2 used 64 GH200 nodes
# with one GPU each and 8 records per GPU (global batch 512); any GPU count that divides 512 gives the same schedule.
#
# usage:  OUT=<output dir> [PY=python] [GPUS_PER_NODE=1] [FM1=<dir with FM1's config.json + model.safetensors>] \
#         [RESUME_FROM=<OUT>/gen/checkpoint_step_NNNNN] bash examples/FM2/train_fm2.sh
# Data: the MaterialsSaddles v2 release under $MATERIALSSADDLES_ROOT (examples/MaterialsSaddles/data_prep.py).
# FM1: downloaded from Hugging Face (SciLM/SaddleFlow, revision f003c26, gated: accept its terms and log in) unless FM1 is set.
set -euo pipefail
: "${OUT:?set OUT to the output directory}" "${MATERIALSSADDLES_ROOT:?set MATERIALSSADDLES_ROOT to the v2 data}"
PY=${PY:-python}; GPUS_PER_NODE=${GPUS_PER_NODE:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,); NN=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD=$(( NN * GPUS_PER_NODE ))
(( 512 % WORLD == 0 )) || { echo "the GPU count ($WORLD) must divide the global batch 512"; exit 1; }
BS=$(( 512 / WORLD ))
STEPS=132226                                    # one epoch of 67.7 M records at batch 512 (sets the EMA window)
EMA=$($PY -c "print(1 - 50 / $STEPS)")          # 0.9996218595435088, the EMA decay FM2 used
mkdir -p "$OUT"
if [ -n "${RESUME_FROM:-}" ]; then
  START="--resume-from $RESUME_FROM"            # the checkpoint holds weights, optimizer, schedule and EMA
else
  if [ -z "${FM1:-}" ]; then
    FM1=$($PY -c "from huggingface_hub import snapshot_download as d; print(d('SciLM/SaddleFlow', revision='f003c2699ac1b9eb0446fa128f9f34c249af9757', allow_patterns=['config.json', 'model.safetensors']))")
  fi
  START="--init-weights $FM1"
fi
# The settings FM2 ran with, resolved: launch_full_slot_d.sh's base command overridden by start_ft.sh 19 (provenance/).
ARGS="--all-subsets --subset-repeat oc22=8,mp20bat=8,oc20=2 --task-name-map oc22=oc20 --com-symmetric-loss \
  --output-dir $OUT/gen --backbone uma-m-1p1 --mixed-precision no --grad-clip-norm 1.0 --batch-size $BS --num-epochs 1 \
  --learning-rate 1e-3 --uma-lr 3e-3 --warmup-steps $(( 250 * WORLD )) --ema-decay $EMA --loss-type huber --huber-delta 0.05 \
  --delta-endpoint-channels 0 --no-inject-force --no-dimer-residual --no-frozen-force-backbone --no-endpoint-features \
  --eigenmode-aux-weight 0 --unfreeze-uma-all --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --head-depth 3 --attn-layers 0 \
  --xt-target-correction --xt-perturb-sigma 0.05 \
  --mover-start-prob 1 --mover-thresh 0.5 --mover-sigma 0.3 --mover-u0-prob 0.5 --inject-flag \
  --mover-all-prob 0.2 --mover-thresh-max 1.0 --mover-ball-prob 0.2 --mover-ball-rmin 3 --mover-ball-rmax 5 --align-com \
  --log-every 50 --save-every-steps 2000 --val-every-steps 2000 --max-val-records 4000 --num-workers 4 $START"
echo "[fm2] $NN nodes x $GPUS_PER_NODE GPUs, $BS records per GPU, EMA $EMA -> $OUT  $(date)"
srun --nodelist="$NODES" --nodes=$NN --ntasks=$WORLD --ntasks-per-node=$GPUS_PER_NODE --unbuffered bash -c "
  export PYTHONPATH=$REPO PYTHONUNBUFFERED=1 MATERIALSSADDLES_ROOT=$MATERIALSSADDLES_ROOT
  export MASTER_ADDR=${NODES%%,*} MASTER_PORT=${MASTER_PORT:-29511} RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
  exec $PY -u $REPO/examples/MaterialsSaddles/train.py $ARGS" > "$OUT/gen.log" 2>&1
echo "[fm2] done $(date)"
