# environment for MaterialsSaddles scripts on a single GPU node (as the ENVS of launch_oc22_64.sh)
G=/scratch/00405/graeme/saddleflow/campaigns/2026-10-15_qsweep; CACHE=$G/caches
source $G/scripts/env.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/00405/graeme/saddleflow/code_snap_1024/SaddleFlow PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export MATERIALSSADDLES_ROOT=/scratch/08405/ilgar/saddleflow/data/MaterialsSaddles_v2
export FAIRCHEM_CACHE_DIR=$CACHE/fairchem_cache HF_HOME=$CACHE/hf HF_HUB_CACHE=$CACHE/hf/hub
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor XDG_CACHE_HOME=$CACHE/xdg
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token CC=gcc CXX=g++
