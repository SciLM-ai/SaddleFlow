#!/bin/bash
#SBATCH --job-name=lic_recovery_mmf
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_recovery_mmf_%j.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_recovery_mmf_%j.err

# Michael: one fixed fold, matched continuations, autonomous evaluation + audit.
# sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh
# Complement: sbatch --export=ALL,LIC_REC_FOLD=1 examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh
# Learnability: sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation learnability
# Matched six-source R2: sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation learnability --mmf
# Recovery horizon: sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation recovery-horizon --r2-ckpt PATH
# Inference-only endpoint kick: sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation sella-restart
# Cross-basin inference: sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh --ablation generalization --starts false-bridges --basins all
set -euo pipefail
LIC_REC_ABLATION="basin-disjoint"
LIC_REC_EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ablation)
            if [[ $# -lt 2 ]]; then echo "Michael: --ablation needs a value" >&2; exit 2; fi
            LIC_REC_ABLATION="$2"; shift 2 ;;
        --ablation=*) LIC_REC_ABLATION="${1#*=}"; shift ;;
        --help|-h)
            echo "Michael: sbatch examples/LiC/experiments/pes_and_recovery/run_recovery_mmf.sh [--ablation basin-disjoint|learnability|generalization|recovery-horizon|sella-restart]"
            echo "Michael: ablation options are forwarded to the corresponding Python script."
            echo "Michael: learnability --mmf adds the detached MMF directional teacher to the same six-source training run."
            echo "Michael: generalization --starts false-bridges --basins all|unseen|1,2 tests exact/noisy false bridges without training."
            echo "Michael: add --sigmas none for exact starts only; default sigmas are .02,.05,.10 A."
            echo "Michael: recovery-horizon --r2-ckpt PATH compares matched R1/R2 at the saved trajectory times, without training."
            echo "Michael: sella-restart uses the original checkpoint, one endpoint kick + fresh flow, and full Sella on ALL original starts."
            exit 0 ;;
        *) LIC_REC_EXTRA+=("$1"); shift ;;
    esac
done
if [[ "$LIC_REC_ABLATION" != "basin-disjoint" && "$LIC_REC_ABLATION" != "learnability" && "$LIC_REC_ABLATION" != "generalization" && "$LIC_REC_ABLATION" != "recovery-horizon" && "$LIC_REC_ABLATION" != "sella-restart" ]]; then
    echo "Michael: unknown ablation '$LIC_REC_ABLATION'" >&2; exit 2
fi
if [[ "$LIC_REC_ABLATION" == "basin-disjoint" && ${#LIC_REC_EXTRA[@]} -gt 0 ]]; then
    echo "Michael: extra CLI options require learnability/generalization/recovery-horizon/sella-restart; use LIC_REC_* exports for the original experiment" >&2
    exit 2
fi
LIC_REC_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_REC_PYTHON="${LIC_REC_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_REC_OUT="${LIC_REC_OUT:-/scratch/gqs7zr/saddleflow/lic_recovery_mmf_${SLURM_JOB_ID:?Submit with sbatch}}"
LIC_REC_INIT="${LIC_REC_INIT:-/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/checkpoint_final}"
LIC_REC_ARCHIVE="${LIC_REC_ARCHIVE:-/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz}"
LIC_REC_FOLD="${LIC_REC_FOLD:-0}"
if [[ "$LIC_REC_ABLATION" == "learnability" ]]; then
    LIC_REC_UPDATES="${LIC_REC_UPDATES:-6000}"
else
    LIC_REC_UPDATES="${LIC_REC_UPDATES:-2000}"
fi
cd "$LIC_REC_REPO"
mkdir -p "$LIC_REC_OUT"
export PYTHONPATH="$LIC_REC_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_REC_OUT/matplotlib_cache"
echo "[batch] Michael: job=$SLURM_JOB_ID ablation=$LIC_REC_ABLATION fold=$LIC_REC_FOLD output=$LIC_REC_OUT"
if [[ "$LIC_REC_ABLATION" == "sella-restart" ]]; then
    srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/sella_restart.py \
        --out-dir "$LIC_REC_OUT" --ckpt "$LIC_REC_INIT" --device cuda "${LIC_REC_EXTRA[@]}"
    exit 0
fi
if [[ "$LIC_REC_ABLATION" == "recovery-horizon" ]]; then
    srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/recovery_horizon.py \
        --out-dir "$LIC_REC_OUT" --device cuda "${LIC_REC_EXTRA[@]}"
    exit 0
fi
if [[ "$LIC_REC_ABLATION" == "generalization" ]]; then
    srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/evaluate_recovery_learnability.py \
        --out-dir "$LIC_REC_OUT" --device cuda "${LIC_REC_EXTRA[@]}"
    exit 0
fi
if [[ "$LIC_REC_ABLATION" == "learnability" ]]; then
    srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/train_recovery_learnability.py \
        --run-dir "$LIC_REC_OUT" --init-weights "$LIC_REC_INIT" --updates "$LIC_REC_UPDATES" \
        --device cuda "${LIC_REC_EXTRA[@]}"
    srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/plot_recovery_learnability.py "$LIC_REC_OUT/learnability.npz"
    exit 0
fi
LIC_REC_SHARED=(--run-dir "$LIC_REC_OUT" --init-weights "$LIC_REC_INIT" --shell-archive "$LIC_REC_ARCHIVE"
                --fold "$LIC_REC_FOLD" --updates "$LIC_REC_UPDATES" --device cuda)
srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/train_recovery_mmf.py "${LIC_REC_SHARED[@]}" --stage prepare
srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/train_recovery_mmf.py "${LIC_REC_SHARED[@]}" --stage calibrate
for LIC_REC_EXPERIMENT in R0 R1 R2; do
    srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/train_recovery_mmf.py "${LIC_REC_SHARED[@]}" \
        --stage train --experiment "$LIC_REC_EXPERIMENT"
done
srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/diagnose_recovery_mmf.py --run-dir "$LIC_REC_OUT" --device cuda
LIC_REC_EVAL_EXTRA=()
if [[ -n "${LIC_REC_FRESH_ARCHIVE:-}" ]]; then
    LIC_REC_EVAL_EXTRA=(--fresh-shell-archive "$LIC_REC_FRESH_ARCHIVE")
fi
srun --unbuffered "$LIC_REC_PYTHON" -u examples/LiC/experiments/pes_and_recovery/compare_recovery_mmf.py \
    --run-dir "$LIC_REC_OUT" --updates "$LIC_REC_UPDATES" --device cuda "${LIC_REC_EVAL_EXTRA[@]}"
