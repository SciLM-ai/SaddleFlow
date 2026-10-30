#!/bin/bash
#SBATCH --job-name=hint_ft_armA
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=4-00:00:00
#SBATCH --output=/bigtemp/gqs7zr/saddleflow/hint_finetune_%x_%j.out
#SBATCH --error=/bigtemp/gqs7zr/saddleflow/hint_finetune_%x_%j.err
#SBATCH --no-requeue

# run_finetune.sh for the CS cluster, set up for arm A: the hinted half from the dataset's own saddles
# (--hint-rows dataset), no new Sella runs. Data under /bigtemp (SADDLEFLOW_DATA, read by finetune.py).
# Same global batch as the pilot, 16 pairs: one process per allocated GPU, 16 / GPUs pairs each, no gradient
# accumulation (train() counts max_steps in micro-batches, so accumulation would cut the run or stretch the
# schedule). 4x A100 80GB (cheetah04, the default) or 2x H100 94GB (8 pairs per GPU). Never resume on a
# different GPU count (the schedule is per rank).
#   sbatch examples/MaterialsSaddles/experiments/hint_finetune/run_finetune_cs.sh
#   sbatch --constraint=h100_94gb --gres=gpu:2 ... run_finetune_cs.sh             # 2x H100 94GB
#   FIRST_WINS=1: once running, cancel this user's pending jobs of the same name (submit the A100 and H100
#   variants both with it; whichever starts first runs, and the other is cancelled)
#   RESUME=<checkpoint dir> OUT=<run dir> sbatch ... run_finetune_cs.sh       # continue a run
#   EXTRA="--hint-rows dataset --inversion-prob 0.5" sbatch ... run_finetune_cs.sh   # another arm
#   RUN_NAME=inversion EXTRA="--inversion-prob 0.5" sbatch --job-name=hint_ft_inv ...  # the pilot + inversion
#   (RUN_NAME names the run directory, default armA; give each arm its own --job-name so FIRST_WINS only
#   cancels that arm's twin)
set -euo pipefail
REPO="${SADDLEFLOW_REPO:-/u/gqs7zr/SaddleFlow}"
PYTHON="${SADDLEFLOW_PYTHON:-/u/gqs7zr/.conda/envs/saddleflow/bin/python}"
export SADDLEFLOW_DATA="${SADDLEFLOW_DATA:-/bigtemp/gqs7zr/saddleflow}"
LABELS="${LABELS:-$SADDLEFLOW_DATA/hint_finetune/tierb/labels.json}"
OUT="${OUT:-$SADDLEFLOW_DATA/hint_finetune/runs/${RUN_NAME:-armA}_${SLURM_JOB_ID}}"
EXTRA="${EXTRA:---hint-rows dataset}"
[ -n "${RESUME:-}" ] && EXTRA="$EXTRA --resume-from $RESUME"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Caches on /bigtemp, not the home quota: the released model and UMA (gated: needs a token), fairchem's kernels.
CACHE=/bigtemp/gqs7zr/cache
export HF_HOME="${HF_HOME:-$CACHE/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
# Offline by default: both weight sets are cached under $CACHE, and with a revoked or expired token the Hub
# raises on a gated repo instead of falling back to the cache. HF_HUB_OFFLINE=0 to fetch something new.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE/inductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE/xdg}" FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$CACHE/fairchem}"
export OMP_NUM_THREADS=4
NGPU="${SLURM_GPUS_ON_NODE:?no GPUs allocated (submit with --gres=gpu:N)}"
VISIBLE=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES:-}" | grep -c . || true)
[ "$VISIBLE" = "$NGPU" ] || { echo "SLURM_GPUS_ON_NODE=$NGPU but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2; exit 1; }
(( 16 % NGPU == 0 )) || { echo "$NGPU GPUs do not divide the global batch of 16 pairs" >&2; exit 1; }
PAIRS=$((16 / NGPU))
export MASTER_ADDR=localhost MASTER_PORT=$((30000 + SLURM_JOB_ID % 30000)) WORLD_SIZE=$NGPU
cd "$REPO"
# Fail in seconds, not after the queue wait plus four model loads, if the env or the inputs are incomplete.
"$PYTHON" -c "import safetensors, accelerate, huggingface_hub, fairchem.core, saddleflow" \
  || { echo "missing packages in $PYTHON (pip install -e '$REPO[train,search]')" >&2; exit 1; }
[ "$HF_HUB_OFFLINE" = 1 ] || [ -s "$HF_TOKEN_PATH" ] || { echo "no Hugging Face token at $HF_TOKEN_PATH (hf auth login)" >&2; exit 1; }
[ -s "$LABELS" ] || { echo "no Tier B labels at $LABELS" >&2; exit 1; }
if [ "${FIRST_WINS:-0}" = 1 ]; then
  # After the checks, so a broken setup never cancels the twin. Lowest job id wins a tie (both started at once).
  other=$(squeue -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -h -o %A | grep -vx "$SLURM_JOB_ID" | sort -n | head -1 || true)
  if [ -n "$other" ] && [ "$other" -lt "$SLURM_JOB_ID" ]; then echo "job $other of this name is already running" >&2; exit 0; fi
  scancel -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING || true
  echo "FIRST_WINS: cancelled any pending $SLURM_JOB_NAME jobs"
fi
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  GPUs $NGPU x $PAIRS pairs  SADDLEFLOW_DATA=$SADDLEFLOW_DATA  OUT=$OUT  EXTRA=$EXTRA"
git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'
# Fail the job as soon as any rank fails, stopping the others (they would otherwise wait on it forever).
pids=()
for r in $(seq 0 $((NGPU - 1))); do
  RANK=$r LOCAL_RANK=$r "$PYTHON" -u examples/MaterialsSaddles/experiments/hint_finetune/finetune.py \
    --labels "$LABELS" --output-dir "$OUT" --batch-pairs "$PAIRS" $EXTRA & pids+=($!)
done
for _ in "${pids[@]}"; do
  if ! wait -n; then
    echo "a rank failed; stopping the others (see the .err file)" >&2
    kill "${pids[@]}" 2>/dev/null || true; wait; exit 1
  fi
done
