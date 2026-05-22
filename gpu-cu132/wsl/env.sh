#!/bin/bash
# WSL build environment for FAISS GPU CUDA 13.2
# Source this file before running any build step:
#   source gpu-cu132/wsl/env.sh

export CUDA_HOME=/usr/local/cuda
export MKL_ROOT=/opt/intel/oneapi/mkl/latest

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MKL_ROOT/lib:${LD_LIBRARY_PATH:-}"

export MKL_INCLUDE_DIR="$MKL_ROOT/include"
export MKL_LIB="$MKL_ROOT/lib/libmkl_rt.so"

# Default to all supported CUDA compute capabilities
export CUDA_ARCHS="${CUDA_ARCHS:-75;80;86;89;90;120}"

# Wheel variant suffix: faiss-{FAISS_VARIANT}.  Pattern: {processor}-{cuda-version}
# Examples: gpu-cu132  gpu-cu128  gpu  cpu
# Leave unset to build the canonical upstream "faiss" wheel.
export FAISS_VARIANT="${FAISS_VARIANT:-gpu-cu132}"

# Repo root (WSL mount path) — adjust if your drive letter differs
export FAISS_ROOT=/mnt/f/GitHub/faiss

echo "[env] CUDA_HOME    = $CUDA_HOME"
echo "[env] MKL_ROOT     = $MKL_ROOT"
echo "[env] CUDA_ARCHS   = $CUDA_ARCHS"
echo "[env] FAISS_VARIANT= $FAISS_VARIANT"
echo "[env] FAISS_ROOT   = $FAISS_ROOT"
