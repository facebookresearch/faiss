#!/usr/bin/env bash

set -e

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PREFIX="${CONDA_PREFIX:-$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)}"
OPENMP_PREFIX="${OpenMP_ROOT:-$PREFIX}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"


cmake -S . -B build \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=ON \
  -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -DOpenMP_ROOT="$OPENMP_PREFIX" 

cmake --build build -j "$JOBS" --target faiss swigfaiss faiss_test faiss_metal_test swigfaiss_avx2 swigfaiss_avx512


source test_env/bin/activate

pip install ./build/faiss/python/