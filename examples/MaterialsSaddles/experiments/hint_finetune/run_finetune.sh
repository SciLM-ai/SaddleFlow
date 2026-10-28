#!/bin/bash
#SBATCH --job-name=hint_ft
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/hint_finetune_%x_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/hint_finetune_%x_%j.err
#SBATCH --no-requeue

# finetune.py on one node, one process per GPU, started directly by this script (no srun: on this cluster
# srun's per-task binding hid some of the node's GPUs from ranks 2-3 and failed set_device). Every process sees
# all 4 GPUs and picks its own with LOCAL_RANK; the static env:// rendezvous is the same SPMD setup as run.sh.
# Resubmit with RESUME=<checkpoint dir> to continue; never on a different GPU count (the schedule is per rank).
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LABELS="${LABELS:-/scratch/gqs7zr/saddleflow/hint_finetune/tierb/labels.json}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/hint_finetune/runs/pilot_${SLURM_JOB_ID}}"
EXTRA="${EXTRA:-}"
[ -n "${RESUME:-}" ] && EXTRA="$EXTRA --resume-from $RESUME"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"     # the gated facebook/UMA checkpoints
export OMP_NUM_THREADS=4
NGPU=4
export MASTER_ADDR=localhost MASTER_PORT=$((30000 + SLURM_JOB_ID % 30000)) WORLD_SIZE=$NGPU
mkdir -p "$OUT"
cd "$REPO"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# Fail the job as soon as any rank fails, stopping the others (they would otherwise wait on it forever).
pids=()
for r in $(seq 0 $((NGPU - 1))); do
  RANK=$r LOCAL_RANK=$r "$PYTHON" -u examples/MaterialsSaddles/experiments/hint_finetune/finetune.py \
    --labels "$LABELS" --output-dir "$OUT" $EXTRA & pids+=($!)
done
for _ in "${pids[@]}"; do
  if ! wait -n; then
    echo "a rank failed; stopping the others (see the .err file)" >&2
    kill "${pids[@]}" 2>/dev/null || true; wait; exit 1
  fi
done
