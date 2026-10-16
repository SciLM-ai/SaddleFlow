#!/bin/bash
#SBATCH --job-name=failure_modes
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/failure_mode_analysis_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/failure_mode_analysis_%j.err
#SBATCH --no-requeue

# Landing points of the released unconditional model from test-set minima, labelled by Sella.
#   sbatch examples/MaterialsSaddles/failure_mode_analysis/run_failure_mode_analysis.sh
#   sbatch --export=ALL,N_STARTS=0 ...      # every unique minimum instead of a random 500
# Extra arguments are passed to make_proposals.py.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
DATA="${DATA:-/scratch/gqs7zr/saddleflow/MaterialsSaddles/mp20bat/test/mp20bat_test_0000.aselmdb}"
N_STARTS="${N_STARTS:-500}"
SEED="${SEED:-0}"
TASK="${TASK:-omat}"
OUT="${OUT:-/scratch/gqs7zr/saddleflow/failure_mode_analysis_mp20bat_${SLURM_JOB_ID:?Submit with sbatch}}"
HERE="$REPO/examples/MaterialsSaddles"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$OUT/matplotlib_cache"
mkdir -p "$OUT"
echo "[batch] job=$SLURM_JOB_ID out=$OUT n_starts=$N_STARTS seed=$SEED data=$DATA"

"$PYTHON" -u "$HERE/failure_mode_analysis/make_proposals.py" --data "$DATA" --out-dir "$OUT" \
  --n-starts "$N_STARTS" --seed "$SEED" --task-name "$TASK" "$@"
# The paper's scoring protocol: UMA-S-1.2, fmax 0.01 eV/A, <= 300 steps, delta0 0.1, index check.
"$PYTHON" -u "$HERE/sella_eval.py" --src "$OUT/proposals.traj" --out "$OUT/sella.json" \
  --model uma-s-1p2 --task-name "$TASK" --fmax 0.01 --steps 300 --delta0 0.1 --check-index
"$PYTHON" -u "$HERE/failure_mode_analysis/analyze_failure_modes.py" "$OUT"
