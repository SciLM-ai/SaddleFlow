#!/bin/bash
#SBATCH --job-name=mirror_test
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100|a6000|a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/mirror_test_%x_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/mirror_test_%x_%j.err
#SBATCH --no-requeue

# mirror_test.py on Rivanna, one GPU (run_mirror_test_cs.sh is the CS-cluster version).
# CKPT: a fine-tune checkpoint directory (ema.pt, model.safetensors; config.json one level up), or 'released'.
# OUT defaults to mirror_test/<run name> (the checkpoint's run directory, or 'released').
#   CKPT=released sbatch --export=ALL run_mirror_test.sh
#   CKPT=/scratch/gqs7zr/saddleflow/hint_finetune/runs/pilot_21872158/checkpoint_final sbatch --export=ALL run_mirror_test.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/scratch/gqs7zr/saddleflow}"
CKPT="${CKPT:?Pass CKPT=<checkpoint dir> or CKPT=released}"
NAME=released; [ "$CKPT" = released ] || NAME=$(basename "$(dirname "$CKPT")")
OUT="${OUT:-$SADDLEFLOW_DATA/mirror_test/$NAME}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"     # the gated facebook/UMA checkpoints
export OMP_NUM_THREADS=4
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
nvidia-smi --query-gpu=name --format=csv,noheader
echo "CKPT=$CKPT  OUT=$OUT"
"$PYTHON" -u examples/MaterialsSaddles/experiments/mirror_test/mirror_test.py --ckpt "$CKPT" --out-dir "$OUT" "$@"
