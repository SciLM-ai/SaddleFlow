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
# MODEL: fm1, fm2 or a release directory. DATASET: a directory of .aselmdb shards or a .traj of triplets (default mp20bat test).
# OUT defaults to mirror_test/<model name>. Pass extra arguments after the script (--n_minima 500).
#   export TMPDIR=/tmp
#   MODEL=fm1 sbatch --export=ALL run_mirror_test_cs.sh
#   MODEL=fm2 DATASET=<dir> sbatch --export=ALL run_mirror_test_cs.sh --n_minima 200
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/u/gqs7zr/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/u/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/bigtemp/gqs7zr/saddleflow}"
MODEL="${MODEL:?Pass MODEL=fm1, fm2 or a release directory}"
DATASET="${DATASET:-$SADDLEFLOW_DATA/MaterialsSaddles/mp20bat/test}"
NAME=$(basename "$MODEL")
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
echo "MODEL=$MODEL  DATASET=$DATASET  OUT=$OUT"
"$PYTHON" -u examples/MaterialsSaddles/mirror_test.py --model "$MODEL" --dataset "$DATASET" --out-dir "$OUT" "$@"
