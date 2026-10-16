#!/bin/bash
#SBATCH --job-name=failure_modes_sella
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gqs7zr/saddleflow/failure_mode_sella_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/failure_mode_sella_%A_%a.err
#SBATCH --no-requeue

# Finish the Sella labelling of a run directory whose single job hit its time limit. Each array task
# takes one shard of the landing points that sella.json does not yet hold and writes sella_shard<k>.json;
# analyze_failure_modes.py merges every sella*.json in the directory.
#   sbatch --export=ALL,OUT=<run dir> examples/MaterialsSaddles/failure_mode_analysis/finish_sella.sh
# Every sella*.json already in OUT is skipped; pass TAG=<name> for a further pass so it does not overwrite
# the sella_shard<k>.json files of the previous one.
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:?Pass OUT=<run dir>}"
TASK="${TASK:-omat}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PYTHON" -u examples/MaterialsSaddles/sella_eval.py --src "$OUT/proposals.traj" \
  --out "$OUT/sella_${TAG:-shard}${SLURM_ARRAY_TASK_ID}.json" --skip-done "$OUT"/sella*.json \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards "${SLURM_ARRAY_TASK_COUNT:-8}" \
  --model uma-s-1p2 --task-name "$TASK" --fmax 0.01 --steps 300 --delta0 0.1 --check-index
