#!/bin/bash
# WSL build environment for FAISS GPU (x86_64 / Intel MKL entry path).
# Source this file before running any build step:
#   source gpu-cu/wsl/env.sh

# CUDA version + CUDA_HOME resolution (single source of truth).
# Override the version per-invocation, e.g.:  FAISS_CUDA_VER=13.3 source gpu-cu/wsl/env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)/cuda_env.sh"

# CUDA_HOME is resolved by cuda_env.sh (versioned toolkit if present); keep it.
export MKL_ROOT=/opt/intel/oneapi/mkl/latest

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MKL_ROOT/lib:${LD_LIBRARY_PATH:-}"

export MKL_INCLUDE_DIR="$MKL_ROOT/include"
export MKL_LIB="$MKL_ROOT/lib/libmkl_rt.so"

# Default to all supported CUDA compute capabilities
export CUDA_ARCHS="${CUDA_ARCHS:-75;80;86;89;90;120}"

# Wheel package name: faiss-{FAISS_VARIANT}. CPU arch is carried by the wheel's
# manylinux platform tag (x86_64 here), not the name. A single-GPU-arch build
# (e.g. CUDA_ARCHS=89) additionally gets a -sm89 suffix for identification.
export FAISS_VARIANT="${FAISS_VARIANT:-gpu-${FAISS_CUDA_TAG}$(faiss_sm_suffix)}"

# Repo root (WSL mount path) — adjust if your drive letter differs
export FAISS_ROOT=/mnt/f/GitHub/faiss

echo "[env] CUDA_VER     = $FAISS_CUDA_VER (tag $FAISS_CUDA_TAG)"
echo "[env] CUDA_HOME    = $CUDA_HOME"
echo "[env] MKL_ROOT     = $MKL_ROOT"
echo "[env] CUDA_ARCHS   = $CUDA_ARCHS"
echo "[env] FAISS_VARIANT= $FAISS_VARIANT"
echo "[env] FAISS_ROOT   = $FAISS_ROOT"
