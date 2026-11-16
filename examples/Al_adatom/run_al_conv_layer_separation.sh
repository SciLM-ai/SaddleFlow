#!/bin/bash
# FM2 fine-tuned on the Al(100) + adatom triplets with conv-layer separation, against the plain fine-tune and FM2 as
# released. Rivanna paths. The three models:
#     fm2                    FM2 as released (SciLM/SaddleFlow revision fm2), no fine-tuning
#     plain                  FM2 fine-tuned with train.py's defaults (FM2's recipe, 20 epochs, batch 8, one GPU)
#     conv_layer_separation  the same plus --conv-layer-separation (saddleflow/models/conv_layer_separation.py)
# Both fine-tunes train on all 353 triplets (no held-out set): the test is recall of those saddles from the minimum.
#
# Each model is sampled with sample.py from the minimum, with the same seed, so the same noise draws:
#     movers  every saddle's own moving atoms flagged, N_MOVERS draws each: recall (saddles reached by their own draws)
#     ball    the mobile atoms within 4 A of a random top-layer atom or the adatom, N_BALL draws: distinct saddles
#             reached, hits, fall-backs to the minimum, and endpoints elsewhere
# and scored with score.py (within 0.2 A of a dataset saddle or the minimum; geometry only, no Sella).
#
# What runs:
#     memcheck  ARM=<arm>: train.py for MEMCHECK_STEPS steps in benchmark mode (no checkpoint): step time and peak
#               memory. Submit both with memcheck-submit before the first full run.
#     train     ARM=<arm>: train.py, then check the release's conv_layer_separation key
#     sample    MODEL=<model>: sample.py movers and ball
#     compare   score.py on the three models
#
#     ACCOUNT=<allocation> bash examples/Al_adatom/run_al_conv_layer_separation.sh memcheck-submit
#     ACCOUNT=<allocation> bash examples/Al_adatom/run_al_conv_layer_separation.sh            # everything
#
# Where everything goes (DATA=/scratch/gqs7zr/saddleflow):
#     $DATA/ckpts/al_fm2_ft_<arm>/release                  each fine-tune (checkpoint_final is ~22 GB)
#     $DATA/results/al_conv_layer_separation/<model>/{movers,ball}/endpoints.traj
#     $DATA/results/al_conv_layer_separation/compare.txt   score.py on all three
#     $DATA/results/al_conv_layer_separation/logs/
set -euo pipefail

# ---------------------------------------------------------------- the experiment's settings
EPOCHS=20               # train.py's default (706 records at batch 8: ~1765 steps)
BATCH=8                 # train.py's default
MEMCHECK_STEPS=10
MEMCHECK_WARMUP=3
TRAIN_SEED=42           # train.py's default
N_MOVERS=10             # draws per saddle (README)
N_BALL=2000             # draws (sample.py's example for ball)
SAMPLE_SEED=0
CONSTRAINT=b200
TRAIN_TIME=04:00:00     # memcheck on b200: 1.10 s/step plain, 1.34 separated; ~49-50 GB peak (1780 steps: ~35-40 min)
SAMPLE_TIME=04:00:00

# ---------------------------------------------------------------- where things are
REPO=/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow
PYTHON=/home/gqs7zr/.conda/envs/saddleflow/bin/python
SCRIPT=$REPO/examples/Al_adatom/run_al_conv_layer_separation.sh
DATA=/scratch/gqs7zr/saddleflow
CACHE=/scratch/gqs7zr/cache
FM2=$CACHE/huggingface/hub/models--SciLM--SaddleFlow/snapshots/1d364ab8fc4de2d1d302ca892297df6567d3d90e
RESULTS=$DATA/results/al_conv_layer_separation
MEMCHECK_DIR=$DATA/checks/al_conv_layer_separation/memcheck
ARMS="plain conv_layer_separation"
MODELS="fm2 $ARMS"

export PYTHONPATH=$REPO PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=/tmp
export HF_HOME=$CACHE/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token

ckpt() { echo "$DATA/ckpts/al_fm2_ft_$1"; }
release() { if [ "$1" = fm2 ]; then echo "$FM2"; else echo "$(ckpt "$1")/release"; fi; }

arm_args() {    # arm -> ARGS
    ARGS=(--init "$FM2" --num-epochs $EPOCHS --batch-size $BATCH --seed $TRAIN_SEED)
    case "$1" in
        plain) ;;
        conv_layer_separation) ARGS+=(--conv-layer-separation) ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

# The state dict is the same for every model, so only the config marks the separated one.
check_release() {
    local want=false
    [ "$1" = conv_layer_separation ] && want=true
    if [ "$1" = fm2 ]; then want=absent; fi
    if [ $want = absent ]; then
        ! grep -q '"conv_layer_separation": true' "$2/config.json" || { echo "$2 is separated" >&2; exit 1; }
    elif ! grep -q "\"conv_layer_separation\": $want" "$2/config.json"; then
        echo "$2/config.json does not set conv_layer_separation to $want" >&2
        exit 1
    fi
}


# ---------------------------------------------------------------- memcheck
memcheck() {
    local out=$MEMCHECK_DIR/${SLURM_JOB_ID}_$1
    mkdir -p "$out"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$out/nvidia_smi.txt" &
    local smi=$!
    arm_args "$1"
    "$PYTHON" -u examples/Al_adatom/train.py "${ARGS[@]}" --output-dir "$out/run" --log-every 1 \
        --max-steps $MEMCHECK_STEPS --bench-warmup-steps $MEMCHECK_WARMUP --bench-output "$out/bench.json"
    kill $smi
    rm -rf "$out/run"
    echo "$1: peak GPU memory (nvidia-smi) $(sort -n "$out/nvidia_smi.txt" | tail -1) MiB"
    cat "$out/bench.json"
}


# ---------------------------------------------------------------- train, sample, compare
train_arm() {
    arm_args "$1"
    "$PYTHON" -u examples/Al_adatom/train.py "${ARGS[@]}" --output-dir "$(ckpt "$1")"
    check_release "$1" "$(ckpt "$1")/release"
}

sample_model() {
    local model; model=$(release "$1")
    check_release "$1" "$model"
    "$PYTHON" -u examples/Al_adatom/sample.py --model "$model" --mode movers --n $N_MOVERS --seed $SAMPLE_SEED \
        --out "$RESULTS/$1/movers"
    "$PYTHON" -u examples/Al_adatom/sample.py --model "$model" --mode ball --n $N_BALL --seed $SAMPLE_SEED \
        --out "$RESULTS/$1/ball"
}

compare() {
    {
        echo "Al(100) + adatom: FM2, and FM2 fine-tuned on all 353 triplets (plain, conv_layer_separation)."
        echo "sample.py seed $SAMPLE_SEED: movers $N_MOVERS draws per saddle, ball $N_BALL draws; score.py, 0.2 A."
        echo
        "$PYTHON" examples/Al_adatom/score.py $(for m in $MODELS; do echo "$RESULTS/$m/movers"; done)
        echo
        "$PYTHON" examples/Al_adatom/score.py $(for m in $MODELS; do echo "$RESULTS/$m/ball"; done)
    } > "$RESULTS/compare.txt"
    cat "$RESULTS/compare.txt"
}


# ---------------------------------------------------------------- after compare: symmetry-aware scores, and Sella
# rescore  rescore_symmetric.py on every sample.py run (CPU): hits allowing relabelled atoms and the slab's symmetry
# sella    sella_check.py on SELLA_N ball draws, the same draws for plain and conv_layer_separation (GPU): are the
#          endpoints that match nothing real first-order saddles connected to the minimum, and are they new?
SELLA_N=300
SELLA_TIME=12:00:00
rescore() {
    "$PYTHON" -u examples/Al_adatom/rescore_symmetric.py \
        $(for m in $MODELS; do echo "$RESULTS/$m/movers $RESULTS/$m/ball"; done) --workers 32 --out "$RESULTS/rescore"
}
sella() {
    nvidia-smi --query-gpu=name --format=csv,noheader
    "$PYTHON" -u examples/Al_adatom/sella_check.py --n $SELLA_N --seed 0 --workers 8 --out "$RESULTS/sella_check" \
        --runs $(for a in $ARMS; do echo "$a=$RESULTS/$a/ball"; done)
}


# ---------------------------------------------------------------- submit
common_args() {
    : "${ACCOUNT:?Pass ACCOUNT=<allocation>}"
    mkdir -p "$RESULTS/logs"
    LOGS="--output=$RESULTS/logs/%x_%j.out --error=$RESULTS/logs/%x_%j.err"
    COMMON="--account=$ACCOUNT --partition=gpu --nodes=1 --ntasks=1 --no-requeue --parsable $LOGS"
    GPU="--constraint=$CONSTRAINT --gres=gpu:1 --cpus-per-task=8 --mem=96G"
}

memcheck_submit() {
    common_args
    for arm in $ARMS; do
        echo "$arm: memcheck $(sbatch $COMMON $GPU --time=00:20:00 --job-name=al_memcheck_$arm \
            --export=ALL,ARM=$arm "$SCRIPT" memcheck)"
    done
}

submit() {
    common_args
    local samples=() tj sj
    sj=$(sbatch $COMMON $GPU --time=$SAMPLE_TIME --job-name=al_sample_fm2 --export=ALL,MODEL=fm2 "$SCRIPT" sample)
    samples+=($sj); echo "fm2: sample $sj"
    for arm in $ARMS; do
        tj=$(sbatch $COMMON $GPU --time=$TRAIN_TIME --job-name=al_train_$arm --export=ALL,ARM=$arm "$SCRIPT" train)
        sj=$(sbatch $COMMON $GPU --time=$SAMPLE_TIME --job-name=al_sample_$arm --dependency=afterok:$tj \
            --export=ALL,MODEL=$arm "$SCRIPT" sample)
        samples+=($sj); echo "$arm: train $tj   sample $sj"
    done
    echo "compare $(sbatch --account=$ACCOUNT --partition=standard --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
        --time=00:30:00 --no-requeue --parsable $LOGS --job-name=al_compare \
        --dependency=afterok:$(IFS=:; echo "${samples[*]}") "$SCRIPT" compare)   results: $RESULTS"
}


checks_submit() {
    common_args
    echo "rescore $(sbatch --account=$ACCOUNT --partition=standard --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G \
        --time=02:00:00 --no-requeue --parsable $LOGS --job-name=al_rescore "$SCRIPT" rescore)"
    echo "sella $(sbatch $COMMON --constraint="b200|rtxpro6000|a100_80gb" --gres=gpu:1 --cpus-per-task=16 --mem=96G \
        --time=$SELLA_TIME --job-name=al_sella_check "$SCRIPT" sella)"
}


# ---------------------------------------------------------------- which stage this call runs
cd "$REPO"
provenance() { git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'; }
case "${1:-submit}" in
    memcheck-submit) memcheck_submit ;;
    memcheck)        provenance; : "${ARM:?}"; memcheck "$ARM" ;;
    submit)          submit ;;
    train)           provenance; : "${ARM:?}"; train_arm "$ARM" ;;
    sample)          provenance; : "${MODEL:?}"; sample_model "$MODEL" ;;
    compare)         compare ;;
    checks-submit)   checks_submit ;;
    rescore)         provenance; rescore ;;
    sella)           provenance; sella ;;
esac
