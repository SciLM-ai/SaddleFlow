#!/bin/bash
# Does gating W2, rather than zeroing it, keep mirror equivariance and recover the precision W2 = 0 lost? FM2
# fine-tuned on oc22 with W2 gates. Rivanna paths. The companion of run_fm2_w2_oc22.sh, whose models are the controls.
#
# W2 gating (saddleflow/models/w2_gate.py): every m > 0 SO(2) layer keeps its W2 path, multiplied by a learned
# per-bond pseudoscalar, which makes the backbone mirror-equivariant by construction like W2 = 0 does, but leaves W2
# usable wherever a bond's surroundings are handed. The gate starts at zero, so training starts from FM2 with W2 = 0,
# the same starting point as the w2 model.
#
# The four models (only w2_gated is trained here; the others come from run_fm2_w2_oc22.sh and its 300-minima
# evaluation):
#     fm2       FM2 as released, no fine-tuning
#     plain     FM2 fine-tuned on oc22
#     w2        FM2 with W2 = 0, fine-tuned on oc22 (--parity-even)
#     w2_gated  FM2 with W2 gates, fine-tuned on oc22 (--w2-gate-width is the only difference from plain)
#
# Everything else is as in run_fm2_w2_oc22.sh: the same recipe, the same training seed (42, train.py's default,
# which plain and w2 used), the same tests with the same seed (0), so the same minima and starting noise.
#
# What runs:
#     train     w2_gated: the plain recipe plus --w2-gate-width, then export to <ckpt>/release
#     test      mirror_test.py (200 minima), then eval_connected_uncond.py (300 minima x 5), on the oc22 test split
#     compare   compare_connected.py: w2_gated against fm2, plain and w2, and the four mirror reports
#
# Run it once, from anywhere; it submits every job, each waiting for what it needs. ACCOUNT has no default: name the
# allocation for each run.
#     ACCOUNT=<allocation> bash examples/MaterialsSaddles/run_fm2_w2_gated_oc22.sh
#
# Where everything goes:
#     /scratch/gqs7zr/saddleflow/ckpts/fm2_oc22_w2_gated/release                   the fine-tuned model
#     /scratch/gqs7zr/saddleflow/results/fm2_w2_gated_oc22/w2_gated/mirror/report.txt
#     /scratch/gqs7zr/saddleflow/results/fm2_w2_gated_oc22/w2_gated/connected/report.txt, results.csv, results.json
#     /scratch/gqs7zr/saddleflow/results/fm2_w2_gated_oc22/compare.txt            the final comparison
#     /scratch/gqs7zr/saddleflow/results/fm2_w2_gated_oc22/logs/                  every job's .out and .err
set -euo pipefail

# ---------------------------------------------------------------- the experiment's settings
STEPS=10000             # optimizer steps of the fine-tune
GLOBAL_BATCH=32         # records per step: 8 on each of 4 GPUs
LEARNING_RATE=5e-4      # head, backbone and gates
EMA_DECAY=0.995         # 1 - 50 / 10000
TRAIN_SEED=42           # train.py's default: the seed of the plain and w2 fine-tunes
GATE_WIDTH=16           # pseudoscalars per bond in each block's W2 gate
N_MINIMA_MIRROR=200
N_MINIMA_CONNECTED=300
N_EVALS=5               # rollouts per minimum
SEED=0                  # the tests' seed, as for the other three models: the same minima and the same starting noise
SELLA_ETA=5e-3          # Sella's finite-difference step (A)
TRAIN_TIME=2-00:00:00
TEST_TIME=1-00:00:00

# ---------------------------------------------------------------- where things are
REPO=/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow
PYTHON=/home/gqs7zr/.conda/envs/saddleflow/bin/python
SCRIPT=$REPO/examples/MaterialsSaddles/run_fm2_w2_gated_oc22.sh
DATA=/scratch/gqs7zr/saddleflow
CACHE=/scratch/gqs7zr/cache

OC22_TEST=$DATA/MaterialsSaddles/oc22/test
FM2_WEIGHTS=$CACHE/huggingface/hub/models--SciLM--SaddleFlow/snapshots/1d364ab8fc4de2d1d302ca892297df6567d3d90e
CKPT=$DATA/ckpts/fm2_oc22_w2_gated
RESULTS=$DATA/results/fm2_w2_gated_oc22
CONTROLS_CONNECTED=$DATA/results/fm2_w2_oc22_n300     # <model>/connected for fm2, plain, w2 (300 minima x 5)
CONTROLS_MIRROR=$DATA/results/fm2_w2_oc22             # <model>/mirror for fm2, plain, w2 (200 minima)

export MATERIALSSADDLES_ROOT=$DATA/MaterialsSaddles
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TMPDIR=/tmp
export HF_HOME=$CACHE/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token     # the gated facebook/UMA checkpoints


# ---------------------------------------------------------------- train: fine-tune w2_gated, then export it
train() {
    local gpus=4
    mkdir -p "$CKPT"

    # The plain arm's arguments (run_fm2_w2_oc22.sh), plus the seed spelled out and the W2 gates.
    local args=(
        --subsets oc22 --task-name-map oc22=oc20
        --init-weights "$FM2_WEIGHTS" --output-dir "$CKPT"
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
        --seed $TRAIN_SEED --w2-gate-width $GATE_WIDTH
    )

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

    "$PYTHON" -u examples/MaterialsSaddles/export_pretrained.py --ckpt "$CKPT/checkpoint_final" --out "$CKPT/release"
}


# ---------------------------------------------------------------- test: mirror test and connected-saddle evaluation
test_model() {
    local model=$CKPT/release

    "$PYTHON" -u examples/MaterialsSaddles/mirror_test.py --model "$model" --dataset "$OC22_TEST" \
        --n_minima $N_MINIMA_MIRROR --seed $SEED --out-dir "$RESULTS/w2_gated/mirror"

    "$PYTHON" -u examples/MaterialsSaddles/eval_connected_uncond.py --model "$model" --dataset "$OC22_TEST" \
        --n_minima $N_MINIMA_CONNECTED --n_evals $N_EVALS --seed $SEED --eta $SELLA_ETA \
        --out-dir "$RESULTS/w2_gated/connected"
}


# ---------------------------------------------------------------- compare: w2_gated beside the three controls
compare() {
    local c=$CONTROLS_CONNECTED gated=$RESULTS/w2_gated/connected
    for name in fm2 plain w2; do
        if ! cmp -s "$c/$name/connected/minima.traj" "$gated/minima.traj"; then
            echo "w2_gated was evaluated on different minima from $name: not comparable" >&2
            exit 1
        fi
    done
    local compare="$PYTHON examples/MaterialsSaddles/compare_connected.py"
    {
        echo "All four models on the same 300 minima x 5 rollouts (seed 0)."
        echo
        $compare fm2="$c/fm2/connected" plain="$c/plain/connected" w2="$c/w2/connected" w2_gated="$gated"
        echo
        $compare plain="$c/plain/connected" w2_gated="$gated" | sed -n '/^AGAINST/,$p'
        echo
        $compare w2="$c/w2/connected" w2_gated="$gated" | sed -n '/^AGAINST/,$p'
        for name in fm2 plain w2; do
            echo
            cat "$CONTROLS_MIRROR/$name/mirror/report.txt"
        done
        echo
        cat "$RESULTS/w2_gated/mirror/report.txt"
    } > "$RESULTS/compare.txt"
    cat "$RESULTS/compare.txt"
}


# ---------------------------------------------------------------- submit: every job, each after what it needs
submit() {
    : "${ACCOUNT:?Pass ACCOUNT=<allocation>}"
    mkdir -p "$RESULTS/logs" "$CKPT"
    local logs="--output=$RESULTS/logs/%x_%j.out --error=$RESULTS/logs/%x_%j.err"
    local common="--account=$ACCOUNT --partition=gpu --nodes=1 --ntasks=1 --no-requeue --parsable $logs"
    local train_gpus="--constraint=b200|rtxpro6000|a100_80gb --gres=gpu:4 --cpus-per-task=32 --mem=320G --time=$TRAIN_TIME"
    # The controls' 300-minima evaluations ran on rtxpro6000 GPUs.
    local test_gpu="--constraint=rtxpro6000 --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=$TEST_TIME"

    local train_job test_job last
    train_job=$(sbatch $common $train_gpus --job-name=train_w2_gated "$SCRIPT" train)
    test_job=$(sbatch $common $test_gpu --job-name=test_w2_gated --dependency=afterok:$train_job "$SCRIPT" test)
    last=$(sbatch --account=$ACCOUNT --partition=standard --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
        --no-requeue --parsable $logs --job-name=compare_w2_gated --dependency=afterok:$test_job "$SCRIPT" compare)

    echo "train $train_job   test $test_job   compare $last"
    echo "results: $RESULTS"
}


# ---------------------------------------------------------------- which stage this call runs
cd "$REPO"
case "${1:-submit}" in
    submit)  submit ;;
    train)   git log -1 --format="code: %h %s"; git status --short -uno | sed 's/^/  uncommitted: /'; train ;;
    test)    git log -1 --format="code: %h %s"; test_model ;;
    compare) compare ;;
esac
