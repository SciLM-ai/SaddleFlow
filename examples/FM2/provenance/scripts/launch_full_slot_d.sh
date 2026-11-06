#!/bin/bash
# ALL-SUBSETS generator training (FM1's data mix: oc20 x2, oc22 x8, mp20bat x8, lemat x1; oc22 -> oc20 task) on a slot of job JOBID (default 1095531): nodes FROM .. FROM+N-1 of its node list (one GPU each), code
# from the /scratch copy (as launch_oc22_64g.sh). Run on a node of the job.
#   gen:     examples/MaterialsSaddles/train.py, the mover recipe (mover starts, flag, x_t 0.05 + convergent target, CoM-symmetric loss, UMA-S,
#            MSE, uma-lr 1e-2) + EXTRA (later arguments win, e.g. --loss-type huber --huber-delta 0.05 --uma-lr 3e-3);
#            BS records per GPU (global batch BS x N; 16 on 16 nodes = the 8 x 32 of the other runs), EPOCHS, LINE (u0 prob).
#   evalgen: ms_eval_gen2.py (or EVAL_SCRIPT) on CKPT (default OUT/gen/checkpoint_final), PASSES x STEPS, DRAWS, -> OUT/eval_gen<TAG>.
# usage: launch_full_slot.sh gen|evalgen <FROM> <N> <OUTDIR>
set -euo pipefail
STAGE=$1; FROM=$2; N=$3; OUT=$4; JOBID=${JOBID:-1095531}
EPOCHS=${EPOCHS:-30}; BS=${BS:-8}; DRAWS=${DRAWS:-10}
G=/scratch/00405/graeme/saddleflow/campaigns/2026-10-15_qsweep; A=/scratch/00405/graeme/saddleflow/campaigns/2026-10-17_al_scratch
WT=/scratch/00405/graeme/saddleflow/code_snap_1025/SaddleFlow; PY=$G/env/bin/python; CACHE=$G/caches; S=$A/scripts
ALL=($(scontrol show hostnames "$(squeue -j $JOBID -h -o %N)")); NODES=$(IFS=,; echo "${ALL[*]:$FROM:$N}")
TRAIN_RECORDS=${TRAIN_RECORDS:-67700000}       # FM1's mix, approximate (EMA window and step estimate only); SUBSETS/REPEAT: other mixes
export SLURM_JOB_ID=$JOBID SLURM_JOBID=$JOBID
for v in SLURM_JOB_NODELIST SLURM_NODELIST SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_NPROCS \
         SLURM_TASKS_PER_NODE SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_ON_NODE SLURM_STEP_ID SLURM_STEPID \
         SLURM_LOCALID SLURM_PROCID SLURM_NODEID; do unset "$v"; done
ENVS="export PYTHONPATH=$WT PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
  export MATERIALSSADDLES_ROOT=/scratch/08405/ilgar/saddleflow/data/MaterialsSaddles_v2
  export FAIRCHEM_CACHE_DIR=$CACHE/fairchem_cache HF_HOME=$CACHE/hf HF_HUB_CACHE=$CACHE/hf/hub
  export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor XDG_CACHE_HOME=$CACHE/xdg
  export HF_TOKEN_PATH=$HOME/.cache/huggingface/token CC=gcc CXX=g++ OMP_NUM_THREADS=4
  export CUDA_VISIBLE_DEVICES=0 NCCL_TIMEOUT=1800 NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=7"
run() {   # run <port> <command>: one rank per node of the slot
  srun --jobid="$JOBID" --overlap --nodelist="$NODES" --nodes=$N --ntasks=$N --ntasks-per-node=1 --unbuffered bash -c "
    $ENVS
    export MASTER_ADDR=${NODES%%,*} MASTER_PORT=$1 RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
    cd /scratch; $2"
}
mkdir -p "$OUT"
if [ "$STAGE" = gen ]; then
  SUBARG="--all-subsets"; [ -n "${SUBSETS:-}" ] && SUBARG="--subsets $SUBSETS"
  TMARG="--task-name-map ${TASKMAP-oc22=oc20}"; [ -z "${TASKMAP-oc22=oc20}" ] && TMARG=""   # TASKMAP= (empty): none (UMA-S)
  GS=$(( EPOCHS * TRAIN_RECORDS / (BS * N) )); GEMA=$($PY -c "print(1 - ${EMA_WINDOW_FRAC:-50} / $GS)")
  L=""; [ -n "${LINE:-}" ] && L="--mover-u0-prob $LINE"
  echo "[full-slot] gen on nodes $FROM..$((FROM + N - 1)) ($N x $BS) ~$GS steps (EMA $GEMA) $L ${EXTRA:-} -> $OUT  $(date)"
  run ${PORT:-27000} "exec $PY $WT/examples/MaterialsSaddles/train.py $SUBARG --subset-repeat ${REPEAT:-oc22=8,mp20bat=8,oc20=2} $TMARG --com-symmetric-loss \
      --output-dir $OUT/gen --backbone uma-s-1p2 --mixed-precision no --grad-clip-norm 1.0 --batch-size $BS \
      --num-epochs $EPOCHS --learning-rate 1e-3 --uma-lr 1e-2 --warmup-steps $(( 250 * N )) --ema-decay $GEMA \
      --loss-type mse --delta-endpoint-channels 0 --no-inject-force --no-dimer-residual --no-frozen-force-backbone \
      --no-endpoint-features --eigenmode-aux-weight 0 --unfreeze-uma-all --early-time-film-blocks 0,1,2,3 --head-depth 3 \
      --attn-layers 0 --xt-target-correction --xt-perturb-sigma 0.05 --mover-start-prob 1 --mover-base start \
      --mover-thresh 0.5 --mover-sigma 0.3 --inject-flag $L --log-every 50 --save-every-steps 2000 \
      --val-every-steps 2000 --max-val-records 4000 --num-workers 4 ${MAXSTEPS:+--max-steps $MAXSTEPS} ${EXTRA:-}" > $OUT/gen.log 2>&1
  echo "[full-slot] gen done $(date)"
elif [ "$STAGE" = evalgen ]; then
  E=$OUT/eval_gen${TAG:-}
  echo "[full-slot] evalgen on nodes $FROM..$((FROM + N - 1)) (${PASSES:-1} x ${STEPS:-20}, $DRAWS draws) -> $E  $(date)"
  run 0 "exec $PY $S/${EVAL_SCRIPT:-ms_eval_gen2.py} --ckpt ${CKPT:-$OUT/gen/checkpoint_final} --subset oc22 --out $E \
      --draws $DRAWS --passes ${PASSES:-1} --steps ${STEPS:-20} ${EVAL_ARGS:-} --shard \$SLURM_PROCID --nshards \$SLURM_NTASKS" > $E.log 2>&1
  $PY $S/${EVAL_SCRIPT:-ms_eval_gen2.py} --out $E --summary
fi
