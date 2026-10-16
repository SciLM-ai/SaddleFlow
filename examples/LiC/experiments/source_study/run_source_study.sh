#!/bin/bash
#SBATCH --job-name=lic_source_study
#SBATCH --partition=gpu
#SBATCH --constraint=a100|a40|b200|a6000|rtxpro6000
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gqs7zr/saddleflow/lic_source_study_%A_%a.out
#SBATCH --error=/scratch/gqs7zr/saddleflow/lic_source_study_%A_%a.err
#SBATCH --no-requeue

# Prepare the split once BEFORE submitting this four-arm array.
# Every usual train.py CLI argument can be forwarded; arm identity is fixed.
set -euo pipefail
LIC_STUDY_REPO="${SADDLEFLOW_REPO:-/sfs/gpfs/tardis/home/gqs7zr/ai4mat/SaddleFlow}"
LIC_STUDY_PYTHON="${LIC_STUDY_PYTHON:-/home/gqs7zr/.conda/envs/saddleflow/bin/python}"
LIC_STUDY_ROOT="${LIC_STUDY_ROOT:-/scratch/gqs7zr/saddleflow/lic_source_study_seed42}"
LIC_STUDY_ARMS=(n12_uncond n12_source n48_uncond n48_source)
LIC_STUDY_TASK="${SLURM_ARRAY_TASK_ID:?submit with sbatch --array=0-3}"
LIC_STUDY_ARM="${LIC_STUDY_ARMS[$LIC_STUDY_TASK]}"
LIC_STUDY_N=12
if [[ "$LIC_STUDY_ARM" == n48_* ]]; then LIC_STUDY_N=48; fi
LIC_STUDY_SOURCE=--no-source-conditioning
if [[ "$LIC_STUDY_ARM" == *_source ]]; then LIC_STUDY_SOURCE=--source-conditioning; fi
cd "$LIC_STUDY_REPO"
[[ -f "$LIC_STUDY_ROOT/split/split.json" ]] || { echo "run source_study.py prepare first" >&2; exit 1; }
mkdir -p "$LIC_STUDY_ROOT/$LIC_STUDY_ARM"
export PYTHONPATH="$LIC_STUDY_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$LIC_STUDY_ROOT/$LIC_STUDY_ARM/matplotlib_cache"
unset SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST
export SLURM_CPU_BIND=none
echo "[batch] job=$SLURM_JOB_ID arm=$LIC_STUDY_ARM root=$LIC_STUDY_ROOT; Slurm GPU assignment preserved"
srun --cpu-bind=none --unbuffered "$LIC_STUDY_PYTHON" -u examples/LiC/train.py \
  --device cuda --max-steps 96000 --save-every-steps 12000 --save-every-epochs 0 "$@" \
  --study-split "$LIC_STUDY_ROOT/split" --train-saddles "$LIC_STUDY_N" "$LIC_STUDY_SOURCE" \
  --output-dir "$LIC_STUDY_ROOT/$LIC_STUDY_ARM"
