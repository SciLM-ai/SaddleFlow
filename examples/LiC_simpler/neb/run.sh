#!/bin/bash
# Serial SaddleMill CI-NEB for the LiC_simpler training triplet: ONE job, run directly
# on the GPU of the node this script is executed on. config.ini sets
# executorlib = False, so there is no srun, no flux and no executorlib involved.
#
#   bash run.sh          # run (a no-op while the recorded outputs are still present)
#   bash run.sh fresh    # delete the recorded outputs first, then run from scratch
#
# Environment (override as needed; the defaults are the Vista setup this was recorded on):
#   SADDLEMILL_DIR      SaddleMill checkout, put on PYTHONPATH
#   SADDLEMILL_ENV_BIN  bin/ of a Python env with fairchem-core, ase and CUDA torch
#   FAIRCHEM_CACHE_DIR  fairchem model cache (UMA-S-1.2 is fetched into it on first use)
set -euo pipefail
cd "$(dirname "$0")"

SADDLEMILL_DIR=${SADDLEMILL_DIR:-/work/08405/ilgar/vista/SaddleMill}
SADDLEMILL_ENV_BIN=${SADDLEMILL_ENV_BIN:-/work/08405/ilgar/vista/conda_libraries/tsearch/bin}
export FAIRCHEM_CACHE_DIR=${FAIRCHEM_CACHE_DIR:-/scratch/08405/ilgar/.cache/fairchem}
export PATH=$SADDLEMILL_ENV_BIN:$PATH
export PYTHONPATH=$SADDLEMILL_DIR${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
# Vista (GH200) only: the system CUDA libraries the tsearch env's torch expects.
if [ -d /opt/apps/cuda/12.4/targets/sbsa-linux/lib ]; then
    export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

if [ "${1:-}" = fresh ]; then
    rm -rf NEB_trajes NEB_status_csvs NEB_debug_zips traj_files_ordered.json saddlemill.log
fi
if [ -e traj_files_ordered.json ]; then
    echo "Outputs of a previous run are present: SaddleMill will resume and skip the converged job."
    echo "Use 'bash run.sh fresh' to rerun from scratch ('git checkout -- .' restores the recorded outputs)."
fi

echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)  start: $(date)" | tee saddlemill.log
python -u -m saddlemill 2>&1 | tee -a saddlemill.log
echo "end: $(date)" | tee -a saddlemill.log
