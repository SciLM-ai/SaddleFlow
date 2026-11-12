#!/bin/bash
#SBATCH --job-name=pseudotensor_probe
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100|a6000|a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/pseudotensor_probe_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/pseudotensor_probe_%j.err
#SBATCH --no-requeue

# pseudotensor_probe.py on one GPU. MODEL: uma (stock UMA-M-1.1, oc20 by default) or fm2 (SaddleFlow FM2, oc22).
# OUT defaults to $SADDLEFLOW_DATA/pseudotensor_probe/<MODEL>. Extra arguments go to the script (--n-records 2).
#   MODEL=uma sbatch --export=ALL run_pseudotensor_probe.sh
#   MODEL=fm2 sbatch --export=ALL --time=00:30:00 run_pseudotensor_probe.sh --n-records 2
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/scratch/gqs7zr/saddleflow}"
MODEL="${MODEL:?Pass MODEL=uma or MODEL=fm2}"
case "$MODEL" in
  uma) DEFAULT_DATASET="$SADDLEFLOW_DATA/MaterialsSaddles/oc20/test" ;;
  fm2) DEFAULT_DATASET="$SADDLEFLOW_DATA/MaterialsSaddles/oc22/test" ;;
  *) echo "MODEL must be uma or fm2"; exit 1 ;;
esac
DATASET="${DATASET:-$DEFAULT_DATASET}"
OUT="${OUT:-$SADDLEFLOW_DATA/pseudotensor_probe/$MODEL}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"     # the gated facebook/UMA checkpoints
export OMP_NUM_THREADS=4
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
nvidia-smi --query-gpu=name --format=csv,noheader
echo "MODEL=$MODEL  DATASET=$DATASET  OUT=$OUT"
"$PYTHON" -u examples/MaterialsSaddles/experiments/pseudotensor_probe/pseudotensor_probe.py \
  --model "$MODEL" --dataset "$DATASET" --out-dir "$OUT" "$@"
