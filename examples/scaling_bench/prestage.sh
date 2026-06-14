#!/bin/bash
# =============================================================================
# prestage.sh — one-time, single-process setup so the benchmark itself measures
# only steady-state training (no downloads, no cache races inside the timed job).
# Run ONCE per machine before sweep.sh, on a node with outbound internet.
#
#   1. Stages the mp20bat subset (~0.6 GB: 32 .aselmdb shards + parquet splits)
#      under $SCRATCH/MaterialsSaddles/ from HuggingFace.
#   2. Pre-warms the UMA-S-1.2 checkpoint into $FAIRCHEM_CACHE_DIR so ranks don't
#      race to re-download it.
#
# Inputs: SADDLEFLOW_PYTHON, SADDLEFLOW_REPO, FAIRCHEM_CACHE_DIR, HF_TOKEN
# (only needed if the dataset repo is gated), HF_HOME/HF_HUB_CACHE (point these
# at $SCRATCH for big pulls — see CLAUDE.md / memory).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SADDLEFLOW_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [ -z "${SADDLEFLOW_PYTHON:-}" ]; then
    if [ -x /global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python ]; then
        SADDLEFLOW_PYTHON=/global/cfs/cdirs/m1883/ilgar/conda_envs/saddleflow/bin/python
    else
        SADDLEFLOW_PYTHON=python
    fi
fi
PYTHON="$SADDLEFLOW_PYTHON"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH:?\$SCRATCH not set}/fairchem_cache}"

echo "[prestage] python=$PYTHON"
echo "[prestage] SCRATCH=$SCRATCH  FAIRCHEM_CACHE_DIR=$FAIRCHEM_CACHE_DIR"

echo "[prestage] 1/2 staging mp20bat dataset under \$SCRATCH/MaterialsSaddles ..."
"$PYTHON" "$REPO/examples/MP20Bat/data_prep.py" --subset mp20bat

echo "[prestage] 2/2 pre-warming UMA-S-1.2 checkpoint into FAIRCHEM_CACHE_DIR ..."
"$PYTHON" -c "
from saddleflow.utils import load_uma_backbone
load_uma_backbone('uma-s-1p2', device='cpu')
print('  UMA-S-1.2 checkpoint present in FAIRCHEM_CACHE_DIR')
"

echo "[prestage] done — ready to run sweep.sh"
