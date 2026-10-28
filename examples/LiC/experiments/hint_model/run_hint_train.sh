#!/bin/bash
#SBATCH --job-name=lic_hint_train
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_hint_train_%x_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_hint_train_%x_%j.err
#SBATCH --no-requeue

# Fine-tune n12_uncond into a LiC hint model. ARM=input (A1: hint injected into the backbone) or
# ARM=startonly (A2: hint only in the start). Extra arguments go to examples/LiC/train.py.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
ARM="${ARM:?Pass ARM=input|startonly}"
ROOT="${ROOT:-/scratch/gqs7zr/saddleflow/lic_hint_model}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
FLAG=--hint-input; [ "$ARM" = startonly ] && FLAG=--no-hint-input
# 266 pairs / batch 4 = 67 updates per epoch; 24k updates. EMA window ~ updates / 50.
"$PYTHON" -u examples/LiC/train.py --hint-pairs-dir /scratch/gqs7zr/saddleflow/lic_hint_pilot/a3 "$FLAG" \
  --output-dir "$ROOT/$ARM" --max-steps 24000 --num-epochs 400 --batch-size 4 \
  --learning-rate 1e-3 --uma-lr 1e-3 --warmup-steps 100 --ema-decay 0.998 \
  --save-every-steps 8000 --save-every-epochs 0 --log-every 500 "$@"
