#!/bin/bash
# Foundation model on MaterialsSaddles v2 (all four subsets), the S30u2 recipe (unconditional UMA-M-1.1, path starts):
#   one epoch over ALL of lemat, oc20 x2, oc22 x8, mp20bat x8 (67.7M training records); 8 per GPU, 1 GPU per node;
#   backbone LR 3e-3, head LR 1e-3, cosine over the epoch, warmup 250 real optimizer steps; EMA 0.9996
#   (window = steps/50 for 132k steps at global 512); checkpoint + validation (4,000 records) every 2,000 steps.
#   Agreed with the user 2026-09-29.
# usage: launch_foundation.sh <JOBID> <NODELIST> <OUTDIR> [extra train.py args, e.g. --resume-from <ckpt>]
#   SERIALIZE=1 -> TORCH_DISTRIBUTED_DEBUG=DETAIL (the Vista NCCL-hang workaround, ~40 % slower); off by default.
set -euo pipefail
JOBID=$1; NODES=$2; OUT=$3; shift 3
WT=/work/08405/ilgar/vista/codes/SaddleFlow
C=/scratch/08405/ilgar/saddleflow/campaigns/2026-09-29_v2_foundation
PY=/work/08405/ilgar/vista/conda_libraries/saddlegen/bin/python
NN=$(scontrol show hostnames "$NODES" | wc -l)
MASTER_ADDR=$(scontrol show hostnames "$NODES" | head -n1)
MASTER_PORT=${FOUNDATION_PORT:-$(( 20000 + JOBID % 40000 ))}   # FOUNDATION_PORT overrides (54084 was taken on 2026-09-29)
WARM=$(( 250 * NN ))            # accelerate ticks the LR scheduler once per rank per optimizer step
export SLURM_JOB_ID=$JOBID SLURM_JOBID=$JOBID
for v in SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_NPROCS \
         SLURM_TASKS_PER_NODE SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_ON_NODE SLURM_STEP_ID SLURM_STEPID \
         SLURM_LOCALID SLURM_PROCID SLURM_NODEID; do unset "$v"; done
echo "[foundation] $NN nodes ($NODES) master $MASTER_ADDR:$MASTER_PORT warmup $WARM ticks -> $OUT  $(date)"
mkdir -p "$OUT"
srun --jobid="$JOBID" --overlap --nodelist="$NODES" --nodes=$NN --ntasks=$NN --ntasks-per-node=1 --unbuffered bash -c "
  export PYTHONPATH=$WT PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore SCRATCH=/scratch/08405/ilgar
  export MATERIALSSADDLES_ROOT=/scratch/08405/ilgar/saddleflow/data/MaterialsSaddles_v2
  export SADDLEFLOW_MATERIALS_SADDLES_ROOT=$C/data
  export SADDLEFLOW_MATERIALS_SADDLES_INDEX_DIR=/scratch/08405/ilgar/saddleflow/data/v2_index_cache
  export FAIRCHEM_CACHE_DIR=/scratch/08405/ilgar/fairchem_cache HF_HOME=/work/08405/ilgar/vista/huggingface
  export CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
  export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
  export NCCL_TIMEOUT=1800 NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=7 ${SERIALIZE:+TORCH_DISTRIBUTED_DEBUG=DETAIL}
  exec $PY $WT/examples/MaterialsSaddles/train.py --all-subsets --output-dir $OUT \
    --subset-repeat oc22=8,mp20bat=8,oc20=2 --task-name-map oc22=oc20 \
    --backbone uma-m-1p1 --mixed-precision no --grad-clip-norm 1.0 --batch-size 8 --num-epochs 1 \
    --learning-rate 1e-3 --uma-lr 3e-3 --warmup-steps $WARM --ema-decay 0.9996 \
    --loss-type huber --huber-delta 0.05 \
    --delta-endpoint-channels 0 --path-start-prob 1.0 --path-noise-sigma 0.3 --path-u-power 2.0 \
    --log-every 50 --save-every-epochs 1 --save-every-steps 2000 --val-every-steps 2000 --max-val-records 4000 \
    --num-workers 2 --unfreeze-uma-all --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --head-depth 3 --attn-layers 0 \
    --com-symmetric-loss --xt-target-correction --xt-perturb-sigma 0.05 \
    --no-inject-force --no-dimer-residual --no-frozen-force-backbone \
    --no-endpoint-features --eigenmode-aux-weight 0 $*
"
