#!/bin/bash
#SBATCH --job-name=hint_pilot
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/hint_pilot_%x_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/hint_pilot_%x_%A_%a.err
#SBATCH --no-requeue

# One stage of the LiC hint pilot; submit_hint_pilot.sh chains them. MODE is one of
#   a0-sella  Sella from every n12_uncond endpoint (array task = GPU; WORKERS processes share each GPU)
#   mirror    the mirror test (one GPU); EXTRA="--vsym" adds the symmetrized-velocity test
#   a3        the classical hint search (array task = GPU; WORKERS processes share each GPU)
#   analyze   both analyses and the combined report (CPU)
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
OUT="${OUT:?Pass OUT=<output dir>}"
MODE="${MODE:?Pass MODE=a0-sella|mirror|a3|analyze}"
WORKERS="${WORKERS:-4}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/gqs7zr/cache/huggingface}"
export OMP_NUM_THREADS=2
PILOT=examples/LiC/experiments/hint_pilot/hint_pilot.py
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS=$(( ${SLURM_ARRAY_TASK_COUNT:-1} * WORKERS ))

case "$MODE" in
  a0-sella)
    for w in $(seq 0 $((WORKERS - 1))); do
      s=$(( TASK * WORKERS + w ))
      # Shards are fixed, so a shard's finished endpoints are all in its own file.
      SKIP=(); [ -f "$OUT/a0/sella_$s.json" ] && SKIP=(--skip-done "$OUT/a0/sella_$s.json")
      "$PYTHON" -u examples/MaterialsSaddles/sella_eval.py --src "$OUT/a0/endpoints.traj" \
        --out "$OUT/a0/sella_$s.json" --shard "$s" --nshards "$NSHARDS" "${SKIP[@]}" \
        --model uma-s-1p2 --task-name omat --fmax 0.005 --steps 300 --delta0 0.1 --check-index --nev 3 &
    done
    wait ;;
  a3)
    for w in $(seq 0 $((WORKERS - 1))); do
      "$PYTHON" -u "$PILOT" a3 --out-dir "$OUT" --shard $(( TASK * WORKERS + w )) --nshards "$NSHARDS" &
    done
    wait ;;
  mirror)
    "$PYTHON" -u "$PILOT" mirror --out-dir "$OUT" ${EXTRA:-} ;;
  analyze)
    "$PYTHON" -u "$PILOT" analyze-a0 --out-dir "$OUT"
    "$PYTHON" -u "$PILOT" analyze-a3 --out-dir "$OUT"
    "$PYTHON" -u "$PILOT" report --out-dir "$OUT" ;;
esac
