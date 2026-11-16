#!/bin/bash
# LiC trained with FM2's pipeline from pretrained UMA-S-1.2, two arms that differ only in --conv-layer-separation:
#     plain                  FM2's recipe (examples/MaterialsSaddles/train.py) on LiC's 12 training triplets
#     conv_layer_separation  the same plus --conv-layer-separation (saddleflow/models/conv_layer_separation.py)
# Both are evaluated like the oc22 fine-tunes: mirror_test.py and eval_connected_uncond.py on LiC's 167 test triplets
# (all 63 unique minima, 5 rollouts each, seed 0: the same minima and starting noise for both arms).
#
# The recipe is FM2's (examples/FM2/README.md) with what LiC needs changed: UMA-S-1.2 instead of FM1/UMA-M, time-FiLM at
# all 4 blocks, backbone LR 1e-2 (UMA-S's optimum; 3e-3 is UMA-M's), the omat task, one GPU x 4 records, and the length
# of the LiC reference run (examples/LiC/README.md: 96,000 steps at 4 records). Starts are FM2's: from the minimum or
# along the line to the saddle, noise on the movers (on LiC only the Li moves), with the flag input. The data are LiC's
# triplets as MaterialsSaddles shards (make_shards.py). LiC epochs are 6 steps, so the end-of-epoch checkpoint and
# validation are off; validation runs every VAL_EVERY steps on the test triplets (monitoring only) and only the final
# checkpoint is kept (~2 GB).
#
# What runs:
#     shards    make_shards.py, once (CPU)
#     smoke     both arms for SMOKE_STEPS steps, export, and a 3-minimum mirror test and evaluation; reports s/step and
#               peak memory; then deletes its checkpoints (keeps logs and reports)
#     train     one arm (ARM=plain or ARM=conv_layer_separation): train, then export to <ckpt>/release
#     test      one arm: mirror_test.py, then eval_connected_uncond.py
#     compare   compare_connected.py on the two arms, and both mirror reports
#     throws    LiC's grid-throw figure (throw_release.py, plot_throws.py) for both arms: $RESULTS/throws.pdf
#               (submitted on its own: AFTER=<train job>:<train job> ACCOUNT=<allocation> bash <this script> throws-submit)
#
#     ACCOUNT=<allocation> bash examples/LiC/fm2_recipe/run_lic_conv_layer_separation.sh smoke-submit
#     ACCOUNT=<allocation> bash examples/LiC/fm2_recipe/run_lic_conv_layer_separation.sh            # both arms
#
# Where everything goes (DATA=/scratch/gqs7zr/saddleflow):
#     $DATA/LiC_fm2_recipe/shards/                       the shards and split manifest
#     $DATA/ckpts/lic_fm2_recipe_<arm>/release           each arm's model
#     $DATA/results/lic_fm2_recipe/<arm>/{mirror,connected}/report.txt
#     $DATA/results/lic_fm2_recipe/compare.txt           the comparison
#     $DATA/results/lic_fm2_recipe/logs/                 every job's .out and .err
set -euo pipefail

# ---------------------------------------------------------------- the experiment's settings
STEPS=96000             # optimizer steps (the LiC reference run's length)
BATCH=4                 # records per step, one GPU
SMOKE_STEPS=300
LEARNING_RATE=1e-3      # head and FiLM
UMA_LR=1e-2             # backbone
EMA_DECAY=0.9995        # 1 - 50 / 96000, rounded
WARMUP=250
VAL_EVERY=8000
TRAIN_SEED=42
N_EVALS=5               # rollouts per minimum
SEED=0
SELLA_ETA=5e-3
TRAIN_TIME=1-00:00:00
TEST_TIME=06:00:00
TRAIN_CONSTRAINT="b200|rtxpro6000|a100_80gb"

# ---------------------------------------------------------------- where things are
REPO=/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow
PYTHON=/home/gqs7zr/.conda/envs/saddleflow/bin/python
SCRIPT=$REPO/examples/LiC/fm2_recipe/run_lic_conv_layer_separation.sh
DATA=/scratch/gqs7zr/saddleflow
CACHE=/scratch/gqs7zr/cache
SHARDS=$DATA/LiC_fm2_recipe/shards
LIC_TEST=$REPO/examples/LiC/test_set.traj
RESULTS=$DATA/results/lic_fm2_recipe
SMOKE_DIR=$DATA/checks/lic_fm2_recipe_smoke
ARMS="plain conv_layer_separation"

export MATERIALSSADDLES_ROOT=$DATA/LiC_fm2_recipe       # train.py looks for dataset_stats_lic.json here
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=/tmp
export HF_HOME=$CACHE/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token     # the gated facebook/UMA checkpoints

ckpt() { echo "$DATA/ckpts/lic_fm2_recipe_$1"; }


# ---------------------------------------------------------------- train
# ARGS for arm $1, $2 steps, output directory $3.
train_args() {
    ARGS=(
        --subsets lic --shards-dir "$SHARDS" --split-manifest "$SHARDS/split_manifest.csv" --default-task-name omat
        --output-dir "$3" --max-steps "$2" --num-epochs 1000000 --batch-size $BATCH
        --learning-rate $LEARNING_RATE --uma-lr $UMA_LR --warmup-steps $WARMUP --ema-decay $EMA_DECAY
        --backbone uma-s-1p2 --mixed-precision no --grad-clip-norm 1.0 --loss-type huber --huber-delta 0.05
        --delta-endpoint-channels 0 --no-inject-force --no-dimer-residual --no-frozen-force-backbone
        --no-endpoint-features --eigenmode-aux-weight 0 --unfreeze-uma-all
        --early-time-film-blocks 0,1,2,3 --head-depth 3 --attn-layers 0
        --com-symmetric-loss --xt-target-correction --xt-perturb-sigma 0.05
        --mover-start-prob 1 --mover-thresh 0.5 --mover-sigma 0.3 --mover-u0-prob 0.5 --inject-flag
        --mover-all-prob 0.2 --mover-thresh-max 1.0 --mover-ball-prob 0.2 --mover-ball-rmin 3 --mover-ball-rmax 5
        --align-com --log-every 500 --save-every-steps 0 --save-every-epochs 1000000
        --val-every-steps $VAL_EVERY --val-every-epochs 1000000 --max-val-records 400 --num-workers 2
        --seed $TRAIN_SEED
    )
    case "$1" in
        plain) ;;
        conv_layer_separation) ARGS+=(--conv-layer-separation) ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

train_arm() {   # arm, steps, output directory, extra train.py arguments
    mkdir -p "$3"
    train_args "$1" "$2" "$3"
    "$PYTHON" -u examples/MaterialsSaddles/train.py "${ARGS[@]}" "${@:4}"
    "$PYTHON" -u examples/MaterialsSaddles/export_pretrained.py --ckpt "$3/checkpoint_final" --out "$3/release"
    check_release "$1" "$3/release"
}

# The state dict is the same for both arms, so only the config marks the separated model.
check_release() {
    local want=false
    [ "$1" = conv_layer_separation ] && want=true
    if ! grep -q "\"conv_layer_separation\": $want" "$2/config.json"; then
        echo "$2/config.json does not set conv_layer_separation to $want" >&2
        exit 1
    fi
}

test_arm() {    # arm, release, output directory, number of minima (0 = all)
    check_release "$1" "$2"
    "$PYTHON" -u examples/MaterialsSaddles/mirror_test.py --model "$2" --dataset "$LIC_TEST" \
        --n_minima "$4" --seed $SEED --out-dir "$3/mirror"
    "$PYTHON" -u examples/MaterialsSaddles/eval_connected_uncond.py --model "$2" --dataset "$LIC_TEST" \
        --n_minima "$4" --n_evals $N_EVALS --seed $SEED --eta $SELLA_ETA --out-dir "$3/connected"
}


# ---------------------------------------------------------------- smoke: short runs of both arms, end to end
smoke() {
    local out=$SMOKE_DIR/$SLURM_JOB_ID
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    for arm in $ARMS; do
        echo "=== $arm"
        mkdir -p "$out/$arm"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$out/$arm/nvidia_smi.txt" &
        local smi=$!
        # every log line with the time it was printed, to time the steps between two log lines
        train_arm "$arm" $SMOKE_STEPS "$out/$arm/run" --log-every 50 \
            | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done | tee "$out/$arm/train.log"
        kill $smi
        awk -v a=100 -v b=$SMOKE_STEPS -v arm=$arm '$2 == "[train]" && $3 == "ep" && $5 == "step" {
                if ($6 == a) ta = $1; if ($6 == b - 50) tb = $1 }
            END { printf "%s: %.3f s/step (steps %d to %d), so %.1f h for '$STEPS' steps\n", arm, (tb - ta) / (b - 50 - a),
                  a, b - 50, (tb - ta) / (b - 50 - a) * '$STEPS' / 3600 }' "$out/$arm/train.log"
        echo "$arm: peak GPU memory $(sort -n "$out/$arm/nvidia_smi.txt" | tail -1) MiB"
        test_arm "$arm" "$out/$arm/run/release" "$out/$arm" 3
        rm -rf "$out/$arm/run"
    done
}


# ---------------------------------------------------------------- throws: LiC's grid-throw figure for both arms
# throw_release.py (random_throw.py's 64 x 64 grid of Li starts, each arm's own inference schedule), then
# plot_throws.py with both arms on the same sheet.
throws() {
    local files=()
    for arm in $ARMS; do
        check_release "$arm" "$(ckpt "$arm")/release"
        "$PYTHON" -u examples/LiC/fm2_recipe/throw_release.py --model "$(ckpt "$arm")/release" --grid 64 \
            --out-dir "$RESULTS/$arm/throw"
        files+=("$RESULTS/$arm/throw/throw.npz")
    done
    "$PYTHON" -u examples/LiC/plot_throws.py "${files[@]}" --labels $ARMS --out "$RESULTS/throws.pdf"
}


# ---------------------------------------------------------------- compare
compare() {
    local p=$RESULTS/plain c=$RESULTS/conv_layer_separation
    if ! cmp -s "$p/connected/minima.traj" "$c/connected/minima.traj"; then
        echo "the two arms were evaluated on different minima: not comparable" >&2
        exit 1
    fi
    {
        echo "LiC, FM2's recipe from UMA-S-1.2: both arms on the same minima x $N_EVALS rollouts (seed $SEED)."
        echo
        "$PYTHON" examples/MaterialsSaddles/compare_connected.py plain="$p/connected" conv_layer_separation="$c/connected"
        echo
        cat "$p/mirror/report.txt"
        echo
        cat "$c/mirror/report.txt"
    } > "$RESULTS/compare.txt"
    cat "$RESULTS/compare.txt"
}


# ---------------------------------------------------------------- submit
common_args() {
    : "${ACCOUNT:?Pass ACCOUNT=<allocation>}"
    mkdir -p "$RESULTS/logs"
    LOGS="--output=$RESULTS/logs/%x_%j.out --error=$RESULTS/logs/%x_%j.err"
    COMMON="--account=$ACCOUNT --partition=gpu --nodes=1 --ntasks=1 --no-requeue --parsable $LOGS"
}

need_shards() {
    if [ ! -f "$SHARDS/split_manifest.csv" ]; then
        echo "no shards in $SHARDS: run '$SCRIPT shards' first" >&2
        exit 1
    fi
}

smoke_submit() {
    common_args; need_shards
    sbatch $COMMON --constraint="$TRAIN_CONSTRAINT" --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
        --job-name=lic_fm2_recipe_smoke "$SCRIPT" smoke
}

submit() {
    common_args; need_shards
    local train_gpu="--constraint=$TRAIN_CONSTRAINT --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$TRAIN_TIME"
    local test_gpu="--constraint=$TRAIN_CONSTRAINT --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=$TEST_TIME"
    local tests=()
    for arm in $ARMS; do
        local tj sj
        tj=$(sbatch $COMMON $train_gpu --job-name=lic_train_$arm --export=ALL,ARM=$arm "$SCRIPT" train)
        sj=$(sbatch $COMMON $test_gpu --job-name=lic_test_$arm --dependency=afterok:$tj --export=ALL,ARM=$arm "$SCRIPT" test)
        tests+=($sj)
        echo "$arm: train $tj   test $sj"
    done
    local cj
    cj=$(sbatch --account=$ACCOUNT --partition=standard --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
        --no-requeue --parsable $LOGS --job-name=lic_compare --dependency=afterok:$(IFS=:; echo "${tests[*]}") \
        "$SCRIPT" compare)
    echo "compare $cj   results: $RESULTS"
}


# The throw figure, after the two training jobs: AFTER=<job>:<job> (default: start now, the releases must exist).
throws_submit() {
    common_args
    local dep=()
    [ -n "${AFTER:-}" ] && dep=(--dependency=afterok:$AFTER)
    sbatch $COMMON --constraint="$TRAIN_CONSTRAINT" --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
        --job-name=lic_throws "${dep[@]}" "$SCRIPT" throws
}


# ---------------------------------------------------------------- which stage this call runs
cd "$REPO"
provenance() { git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'; }
case "${1:-submit}" in
    shards)       "$PYTHON" -u examples/LiC/fm2_recipe/make_shards.py --out "$SHARDS" ;;
    smoke-submit) smoke_submit ;;
    smoke)        provenance; smoke ;;
    submit)       submit ;;
    train)        provenance; : "${ARM:?}"; train_arm "$ARM" $STEPS "$(ckpt "$ARM")" ;;
    test)         provenance; : "${ARM:?}"; test_arm "$ARM" "$(ckpt "$ARM")/release" "$RESULTS/$ARM" 0 ;;
    compare)      compare ;;
    throws-submit) throws_submit ;;
    throws)       provenance; throws ;;
esac
