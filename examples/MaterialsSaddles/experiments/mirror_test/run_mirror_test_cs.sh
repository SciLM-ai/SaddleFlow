#!/bin/bash
#SBATCH --job-name=mirror_test
#SBATCH --partition=gpu
#SBATCH --constraint=h100_94gb|a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/bigtemp/gqs7zr/saddleflow/mirror_test_%x_%j.out
#SBATCH --error=/bigtemp/gqs7zr/saddleflow/mirror_test_%x_%j.err
#SBATCH --no-requeue

# mirror_test.py on the CS cluster, one GPU (data and caches under /bigtemp, weights offline, as run_eval_cs.sh).
# CKPT: a fine-tune checkpoint directory, or 'released'. OUT defaults to mirror_test/<run name> (the checkpoint's
# run directory, or 'released').
#   export TMPDIR=/tmp
#   CKPT=released sbatch --export=ALL run_mirror_test_cs.sh
#   CKPT=/bigtemp/gqs7zr/saddleflow/hint_finetune/runs/inversion_6751338/checkpoint_final sbatch --export=ALL run_mirror_test_cs.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/u/gqs7zr/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/u/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/bigtemp/gqs7zr/saddleflow}"
CKPT="${CKPT:?Pass CKPT=<checkpoint dir> or CKPT=released}"
NAME=released; [ "$CKPT" = released ] || NAME=$(basename "$(dirname "$CKPT")")
OUT="${OUT:-$SADDLEFLOW_DATA/mirror_test/$NAME}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CACHE=/bigtemp/gqs7zr/cache
export HF_HOME="${HF_HOME:-$CACHE/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE/inductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE/xdg}" FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$CACHE/fairchem}"
export OMP_NUM_THREADS=4
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
nvidia-smi --query-gpu=name --format=csv,noheader
echo "CKPT=$CKPT  OUT=$OUT"
"$PYTHON" -u examples/MaterialsSaddles/experiments/mirror_test/mirror_test.py --ckpt "$CKPT" --out-dir "$OUT" "$@"
