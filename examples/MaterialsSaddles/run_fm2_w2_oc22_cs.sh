#!/bin/bash
# Does W2 = 0 cost precision? FM2 fine-tuned on oc22 with and without W2 = 0. CS cluster paths.
#
# The question: W2 = 0 makes a model mirror-equivariant by construction (saddleflow/utils/parity.py). On mp20bat it did
# so with no loss of precision after training. Does that hold on oc22?
#
# The three models:
#     fm2     FM2 as released, no fine-tuning
#     plain   FM2 fine-tuned on oc22
#     w2      FM2 with W2 = 0, fine-tuned on oc22 (--parity-even is the only difference from plain)
#
# What runs:
#     train     plain and w2: FM2's recipe on oc22, then export to <ckpt>/release
#     test      every model: mirror_test.py, then eval_connected_uncond.py, on the oc22 test split
#     compare   compare_connected.py over the three models
#
# Where everything goes:
#     /bigtemp/gqs7zr/saddleflow/ckpts/fm2_oc22_plain/release      the fine-tuned models
#     /bigtemp/gqs7zr/saddleflow/ckpts/fm2_oc22_w2/release
#     /bigtemp/gqs7zr/saddleflow/results/fm2_w2_oc22/<model>/mirror/report.txt
#     /bigtemp/gqs7zr/saddleflow/results/fm2_w2_oc22/<model>/connected/report.txt, results.csv, results.json
#     /bigtemp/gqs7zr/saddleflow/results/fm2_w2_oc22/compare.txt   the final comparison
#     /bigtemp/gqs7zr/saddleflow/results/fm2_w2_oc22/logs/         every job's .out and .err
#
# Run it once, from anywhere; it submits every job, each waiting for what it needs:
#     bash examples/MaterialsSaddles/run_fm2_w2_oc22_cs.sh
set -euo pipefail

# ---------------------------------------------------------------- the experiment's settings
STEPS=10000             # optimizer steps of each fine-tune
GLOBAL_BATCH=32         # records per step: 8 on each of 4 A100s
LEARNING_RATE=5e-4      # head and backbone
EMA_DECAY=0.995         # 1 - 50 / STEPS
N_MINIMA_MIRROR=200
N_MINIMA_CONNECTED=100
N_EVALS=5               # rollouts per minimum
SEED=0
SELLA_ETA=5e-3          # Sella's finite-difference step (A)

# ---------------------------------------------------------------- where things are
REPO=/u/gqs7zr/SaddleFlow
PYTHON=/u/gqs7zr/.conda/envs/saddleflow/bin/python
SCRIPT=$REPO/examples/MaterialsSaddles/run_fm2_w2_oc22_cs.sh
DATA=/bigtemp/gqs7zr/saddleflow
CACHE=/bigtemp/gqs7zr/cache

OC22_TEST=$DATA/MaterialsSaddles/oc22/test
FM2_WEIGHTS=$CACHE/huggingface/hub/models--SciLM--SaddleFlow/snapshots/1d364ab8fc4de2d1d302ca892297df6567d3d90e
CKPTS=$DATA/ckpts
RESULTS=$DATA/results/fm2_w2_oc22

export MATERIALSSADDLES_ROOT=$DATA/MaterialsSaddles
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME=$CACHE/huggingface HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export XDG_CACHE_HOME=$CACHE/xdg FAIRCHEM_CACHE_DIR=$CACHE/fairchem


# ---------------------------------------------------------------- train: fine-tune one arm, then export it
train() {
    local arm=$1
    local out=$CKPTS/fm2_oc22_$arm
    local gpus=$SLURM_GPUS_ON_NODE
    mkdir -p "$out"

    # FM2's arguments (examples/FM2/train_fm2.sh), on one subset with the fine-tune's steps, batch and rates.
    local args=(
        --subsets oc22 --task-name-map oc22=oc20
        --init-weights "$FM2_WEIGHTS" --output-dir "$out"
        --max-steps $STEPS --num-epochs 1000000 --batch-size $((GLOBAL_BATCH / gpus))
        --learning-rate $LEARNING_RATE --uma-lr $LEARNING_RATE --warmup-steps $((250 * gpus)) --ema-decay $EMA_DECAY
        --backbone uma-m-1p1 --mixed-precision no --grad-clip-norm 1.0 --loss-type huber --huber-delta 0.05
        --delta-endpoint-channels 0 --no-inject-force --no-dimer-residual --no-frozen-force-backbone
        --no-endpoint-features --eigenmode-aux-weight 0 --unfreeze-uma-all
        --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --head-depth 3 --attn-layers 0
        --com-symmetric-loss --xt-target-correction --xt-perturb-sigma 0.05
        --mover-start-prob 1 --mover-thresh 0.5 --mover-sigma 0.3 --mover-u0-prob 0.5 --inject-flag
        --mover-all-prob 0.2 --mover-thresh-max 1.0 --mover-ball-prob 0.2 --mover-ball-rmin 3 --mover-ball-rmax 5
        --align-com --log-every 50 --save-every-steps 5000 --val-every-steps 2500 --max-val-records 4000 --num-workers 4
    )
    if [ "$arm" = w2 ]; then
        args+=(--parity-even)
    fi

    # One process per GPU; stop as soon as any fails (the others would wait on it forever).
    export MASTER_ADDR=localhost MASTER_PORT=$((30000 + SLURM_JOB_ID % 30000)) WORLD_SIZE=$gpus
    local pids=()
    for rank in $(seq 0 $((gpus - 1))); do
        RANK=$rank LOCAL_RANK=$rank "$PYTHON" -u examples/MaterialsSaddles/train.py "${args[@]}" &
        pids+=($!)
    done
    for _ in "${pids[@]}"; do
        if ! wait -n; then
            echo "a rank failed; stopping the others" >&2
            kill "${pids[@]}" 2>/dev/null || true
            exit 1
        fi
    done

    "$PYTHON" -u examples/MaterialsSaddles/export_pretrained.py --ckpt "$out/checkpoint_final" --out "$out/release"
}


# ---------------------------------------------------------------- test: mirror test and connected-saddle evaluation
test_model() {
    local name=$1 model=$2

    "$PYTHON" -u examples/MaterialsSaddles/mirror_test.py --model "$model" --dataset "$OC22_TEST" \
        --n_minima $N_MINIMA_MIRROR --seed $SEED --out-dir "$RESULTS/$name/mirror"

    "$PYTHON" -u examples/MaterialsSaddles/eval_connected_uncond.py --model "$model" --dataset "$OC22_TEST" \
        --n_minima $N_MINIMA_CONNECTED --n_evals $N_EVALS --seed $SEED --eta $SELLA_ETA \
        --out-dir "$RESULTS/$name/connected"
}


# ---------------------------------------------------------------- compare: the three models side by side
compare() {
    {
        "$PYTHON" examples/MaterialsSaddles/compare_connected.py \
            fm2="$RESULTS/fm2/connected" plain="$RESULTS/plain/connected" w2="$RESULTS/w2/connected"
        for name in fm2 plain w2; do
            echo
            cat "$RESULTS/$name/mirror/report.txt"
        done
    } > "$RESULTS/compare.txt"
    cat "$RESULTS/compare.txt"
}


# ---------------------------------------------------------------- submit: every job, each after what it needs
submit() {
    mkdir -p "$RESULTS/logs" "$CKPTS"
    local logs="--output=$RESULTS/logs/%x_%j.out --error=$RESULTS/logs/%x_%j.err"
    local common="--partition=gpu --nodes=1 --ntasks=1 --no-requeue --parsable $logs"
    local train_gpus="--constraint=a100_80gb --gres=gpu:4 --cpus-per-task=32 --mem=320G --time=2-00:00:00"
    local test_gpu="--constraint=h100_94gb|a100_80gb|a100_40gb|a40 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=1-00:00:00"

    local plain w2 test_fm2 test_plain test_w2 last
    plain=$(sbatch $common $train_gpus --job-name=train_plain "$SCRIPT" train plain)
    w2=$(sbatch $common $train_gpus --job-name=train_w2 "$SCRIPT" train w2)

    test_fm2=$(sbatch $common $test_gpu --job-name=test_fm2 "$SCRIPT" test fm2 fm2)
    test_plain=$(sbatch $common $test_gpu --job-name=test_plain --dependency=afterok:$plain \
        "$SCRIPT" test plain "$CKPTS/fm2_oc22_plain/release")
    test_w2=$(sbatch $common $test_gpu --job-name=test_w2 --dependency=afterok:$w2 \
        "$SCRIPT" test w2 "$CKPTS/fm2_oc22_w2/release")

    last=$(sbatch --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
        --no-requeue --parsable $logs --job-name=compare --dependency=afterok:$test_fm2:$test_plain:$test_w2 \
        "$SCRIPT" compare)

    echo "train  plain $plain   w2 $w2"
    echo "test   fm2 $test_fm2   plain $test_plain   w2 $test_w2"
    echo "compare $last"
    echo "results: $RESULTS"
}


# ---------------------------------------------------------------- which stage this call runs
cd "$REPO"
case "${1:-submit}" in
    submit)  submit ;;
    train)   git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'; train "$2" ;;
    test)    git log -1 --format="code: %h %s"; test_model "$2" "$3" ;;
    compare) compare ;;
esac
