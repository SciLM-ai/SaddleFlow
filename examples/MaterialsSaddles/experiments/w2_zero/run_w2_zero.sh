#!/bin/bash
#SBATCH --job-name=w2_zero
#SBATCH --partition=gpu
#SBATCH --constraint=b200|rtxpro6000|a100|a6000|a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/logs/w2_zero_%x_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/logs/w2_zero_%x_%j.err
#SBATCH --no-requeue

# W2 = 0 without retraining, one GPU. CKPT: a fine-tune checkpoint directory, or 'released'.
#   MODE=diagnostic  diagnostic.py (stock vs W2 = 0 velocities and features)  -> w2_zero/<name>/diagnostic
#   MODE=mirror      mirror_test.py with SF_PARITY_EVEN=1                     -> mirror_test/<name>_w2zero
#   MODE=mirror-asis mirror_test.py on the weights as trained (e.g. a --parity-even fine-tune) -> mirror_test/<name>
#   CKPT=released MODE=mirror sbatch run_w2_zero.sh
#   CKPT=/scratch/gqs7zr/saddleflow/hint_finetune/runs/pilot_21872158/checkpoint_final MODE=diagnostic sbatch run_w2_zero.sh
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/scratch/gqs7zr/saddleflow}"
CKPT="${CKPT:?Pass CKPT=<checkpoint dir> or CKPT=released}"
MODE="${MODE:?Pass MODE=diagnostic|mirror|mirror-asis}"
NAME=released; [ "$CKPT" = released ] || NAME=$(basename "$(dirname "$CKPT")")
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TMPDIR=/tmp OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"     # the gated facebook/UMA checkpoints
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
nvidia-smi --query-gpu=name --format=csv,noheader
case "$MODE" in
  diagnostic)
    unset SF_PARITY_EVEN
    OUT="${OUT:-$SADDLEFLOW_DATA/w2_zero/$NAME/diagnostic}"
    echo "CKPT=$CKPT  OUT=$OUT"
    "$PYTHON" -u examples/MaterialsSaddles/experiments/w2_zero/diagnostic.py --ckpt "$CKPT" --out-dir "$OUT" "$@" ;;
  mirror-asis)
    unset SF_PARITY_EVEN
    OUT="${OUT:-$SADDLEFLOW_DATA/mirror_test/$NAME}"
    echo "CKPT=$CKPT  OUT=$OUT  (weights as trained)"
    "$PYTHON" -u examples/MaterialsSaddles/experiments/mirror_test/mirror_test.py --ckpt "$CKPT" --out-dir "$OUT" "$@" ;;
  mirror)
    export SF_PARITY_EVEN=1
    OUT="${OUT:-$SADDLEFLOW_DATA/mirror_test/${NAME}_w2zero}"
    echo "CKPT=$CKPT  OUT=$OUT  SF_PARITY_EVEN=1"
    "$PYTHON" -u examples/MaterialsSaddles/experiments/mirror_test/mirror_test.py --ckpt "$CKPT" --out-dir "$OUT" "$@" ;;
  *) echo "MODE must be diagnostic, mirror or mirror-asis" >&2; exit 1 ;;
esac
