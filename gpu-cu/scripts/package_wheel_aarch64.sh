#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Package FAISS aarch64 / DGX Spark wheel (faiss-gpu-${FAISS_CUDA_TAG}, manylinux aarch64)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAISS_ROOT="${FAISS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$FAISS_ROOT"

# CUDA version (single source of truth — bump in cuda_env.sh for cu133)
source "$SCRIPT_DIR/cuda_env.sh"

PYTHON="${PYTHON:-python3}"
BUILD_OUTPUT_DIR="build_output_aarch64"
# DGX Spark targets a single arch (SM 121), so the package name carries -sm121:
# faiss-gpu-${FAISS_CUDA_TAG}-sm121. The wheel's manylinux_*_aarch64 platform tag
# still selects aarch64 at install time.
export CUDA_ARCHS="121"
FAISS_VARIANT="gpu-${FAISS_CUDA_TAG}$(faiss_sm_suffix)"
PY_VER=$(${PYTHON} -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
BUILD_DIR="_build_python_aarch64_${PY_VER}"

# cuVS-spark
CUVS_REPO="${CUVS_REPO:-/home/zbrad/gh/cuvs}"
CUVS_DIR="${CUVS_DIR:-${CUVS_REPO}/cpp/build}"

export FAISS_VARIANT

echo "========================================="
echo "Packaging FAISS Wheel (DGX Spark)"
echo "========================================="
echo "Package name   : faiss-${FAISS_VARIANT}"
echo "Output directory: $BUILD_OUTPUT_DIR"
echo ""

# Verify build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found. Run build_pkg_aarch64.sh first."
    exit 1
fi

mkdir -p "$BUILD_OUTPUT_DIR"

# Build wheel
echo "[1/3] Building wheel with setuptools..."
cd "$BUILD_DIR"
FAISS_VARIANT="$FAISS_VARIANT" CUDA_ARCHS="121" $PYTHON setup.py bdist_wheel

# Copy wheel to output directory
echo "[2/3] Copying wheel to output..."
wheel_file=$(find dist -name "*.whl" -type f | head -1)
if [ -z "$wheel_file" ]; then
    echo "ERROR: No wheel file found in $BUILD_DIR/dist/"
    exit 1
fi
cp "$wheel_file" "../$BUILD_OUTPUT_DIR/"
wheel_basename=$(basename "$wheel_file")

# Repair wheel: bundle libfaiss-aarch64-${FAISS_CUDA_TAG}.so and libcuvs-spark.so, fix RPATHs
cd ..
if command -v auditwheel &> /dev/null; then
    echo "[3/3] Repairing wheel with auditwheel..."
    export LD_LIBRARY_PATH="${FAISS_ROOT}/_libfaiss_stage_aarch64/lib:${CUVS_DIR}:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH"
    auditwheel repair "$BUILD_OUTPUT_DIR/$wheel_basename" \
        --exclude libcudart.so.13 \
        --exclude libcublas.so.13 \
        --exclude libcublasLt.so.13 \
        --exclude libopenblas.so.0 \
        -w "$BUILD_OUTPUT_DIR/repaired/" 2>&1 | grep -E "INFO|WARNING|ERROR|Fixed"
    repaired_wheel=$(find "$BUILD_OUTPUT_DIR/repaired/" -name "*.whl" | head -1)
    wheel_basename=$(basename "$repaired_wheel")
else
    echo "  (auditwheel not found — skipping wheel repair)"
    repaired_wheel="$BUILD_OUTPUT_DIR/$wheel_basename"
fi

echo ""
echo "========================================="
echo "✓ Wheel packaging complete (DGX Spark)"
echo "========================================="
echo "Wheel: $repaired_wheel"
echo ""
echo "To install on DGX Spark:"
echo "  pip install $repaired_wheel"
