#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Build C++ library (libfaiss) for aarch64 / DGX Spark (SM 121 — GB10 Grace Blackwell)
# Produces libfaiss-aarch64-${FAISS_CUDA_TAG}.so / libfaiss_c-aarch64-${FAISS_CUDA_TAG}.so
# Uses libcuvs-spark from github.com/zbrad/cuvs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAISS_ROOT="${FAISS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$FAISS_ROOT"

# CUDA version (single source of truth — bump in cuda_env.sh for cu133)
source "$SCRIPT_DIR/cuda_env.sh"

# Environment setup
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCHS="121-real"
PYTHON="${PYTHON:-python3}"
FAISS_ENABLE_CUVS="${FAISS_ENABLE_CUVS:-ON}"
BUILD_DIR="_build_aarch64"

# Redirect all output to log file inside build dir
mkdir -p "$BUILD_DIR"
exec > >(tee "$BUILD_DIR/build.log") 2>&1

# cuVS-spark: built from github.com/zbrad/cuvs
CUVS_REPO="${CUVS_REPO:-/home/zbrad/gh/cuvs}"
CUVS_DIR="${CUVS_DIR:-${CUVS_REPO}/cpp/build}"

# WSL: ensure CUDA is on PATH
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "========================================="
echo "Building FAISS C++ Library (DGX Spark)"
echo "========================================="
echo "CUDA_HOME:        $CUDA_HOME"
echo "CUDA_ARCHS:       $CUDA_ARCHS  (SM 121 — GB10 Grace Blackwell)"
echo "FAISS_ENABLE_CUVS: $FAISS_ENABLE_CUVS"
echo "CUVS_DIR:         $CUVS_DIR  (libcuvs-spark)"
echo "BUILD_DIR:        $BUILD_DIR"
echo ""

# Verify CUDA
echo "[1/3] Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please set CUDA_HOME correctly."
    exit 1
fi
echo "CUDA compiler: $(nvcc --version | grep -E 'release|version')"

# Verify libcuvs-spark
echo "[1b] Verifying libcuvs-spark..."
if [[ ! -f "${CUVS_DIR}/libcuvs-spark.so" ]]; then
    echo "ERROR: libcuvs-spark.so not found at ${CUVS_DIR}"
    echo "  Build it first: cd ${CUVS_REPO} && ./build_dgx_spark.sh"
    exit 1
fi
echo "cuVS-spark library: ${CUVS_DIR}/libcuvs-spark.so"

# Verify OpenBLAS (no MKL on aarch64)
OPENBLAS_LIB="${OPENBLAS_LIB:-/usr/lib/aarch64-linux-gnu/libopenblas.so}"
if [[ ! -f "$OPENBLAS_LIB" ]]; then
    echo "ERROR: OpenBLAS not found at $OPENBLAS_LIB"
    echo "  Install with: apt-get install libopenblas-dev"
    exit 1
fi
echo "OpenBLAS library: $OPENBLAS_LIB"

CMAKE_PREFIX_PATH="$CUDA_HOME"

# CMake configuration
echo ""
echo "[2/3] Configuring with CMake..."
rm -rf "$BUILD_DIR"
cmake -B "$BUILD_DIR" \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_CUVS="$FAISS_ENABLE_CUVS" \
    -DFAISS_ENABLE_MKL=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_OPT_LEVEL=generic \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_TOOLKIT_INCLUDE_DIR="$CUDA_HOME/include" \
    -DBLAS_LIBRARIES="$OPENBLAS_LIB" \
    -DLAPACK_LIBRARIES="$OPENBLAS_LIB" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OUTPUT_NAME=faiss-aarch64-${FAISS_CUDA_TAG} \
    -DFAISS_C_OUTPUT_NAME=faiss_c-aarch64-${FAISS_CUDA_TAG} \
    -DFAISS_CUVS_SPARK_LIBRARY="${CUVS_DIR}/libcuvs-spark.so" \
    -Dcuvs_DIR="$CUVS_DIR" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    .

# Build
echo ""
echo "[3/3] Building libraries..."
num_jobs=${FAISS_BUILD_JOBS:-$(nproc)}
echo "Using $num_jobs parallel jobs"

make -C "$BUILD_DIR" -j"$num_jobs" faiss faiss_c

# Stage libraries for next build step
mkdir -p _libfaiss_stage_aarch64/
cmake --install "$BUILD_DIR" --prefix _libfaiss_stage_aarch64/ --config Release

echo ""
echo "========================================="
echo "✓ DGX Spark C++ library build complete"
echo "========================================="
echo "Architecture: SM 121 (GB10 Grace Blackwell)"
echo "cuVS library: libcuvs-spark.so"
echo "Libraries built in: $BUILD_DIR/faiss/"
echo "  libfaiss-aarch64-${FAISS_CUDA_TAG}.so  (main C++ library)"
echo "  libfaiss_c-aarch64-${FAISS_CUDA_TAG}.so  (C API wrapper)"
echo "Staged in: _libfaiss_stage_aarch64/"
