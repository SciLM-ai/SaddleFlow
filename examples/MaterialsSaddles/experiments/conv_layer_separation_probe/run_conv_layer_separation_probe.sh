#!/bin/bash
#SBATCH --job-name=conv_layer_separation_probe
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/conv_layer_separation_probe/probe_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/conv_layer_separation_probe/probe_%j.err
#SBATCH --no-requeue

# conv_layer_separation_probe.py on one GPU: velocity error by flow time for the plain, conv_layer_separation and W2 = 0
# fine-tunes, FM2, and FM2 with separation (step 0); the turn-turn path's size and the reliance test. Forward passes only.
# Release directories and the output default to the oc22 fine-tunes; extra arguments go to the script.
#   sbatch -A <account> run_conv_layer_separation_probe.sh
#   sbatch -A <account> --time=00:30:00 run_conv_layer_separation_probe.sh --n-records 8 --draws 1
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
DATA="${SADDLEFLOW_DATA:-/scratch/gqs7zr/saddleflow}"
CKPTS="$DATA/ckpts"
FM2="${FM2:-/scratch/gqs7zr/cache/huggingface/hub/models--SciLM--SaddleFlow/snapshots/1d364ab8fc4de2d1d302ca892297df6567d3d90e}"
DATASET="${DATASET:-$DATA/MaterialsSaddles/oc22/test}"
OUT="${OUT:-$DATA/conv_layer_separation_probe}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=4
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
nvidia-smi --query-gpu=name --format=csv,noheader
echo "DATASET=$DATASET  OUT=$OUT"
"$PYTHON" -u examples/MaterialsSaddles/experiments/conv_layer_separation_probe/conv_layer_separation_probe.py \
  --dataset "$DATASET" --out-dir "$OUT" \
  --plain "$CKPTS/fm2_oc22_plain/release" \
  --separated "$CKPTS/fm2_oc22_conv_layer_separation/release" \
  --w2 "$CKPTS/fm2_oc22_w2/release" \
  --fm2 "$FM2" "$@"
