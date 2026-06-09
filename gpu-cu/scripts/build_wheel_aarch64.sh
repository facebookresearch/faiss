#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Unified build script for FAISS aarch64 / DGX Spark wheel
# Target: NVIDIA DGX Spark — GB10 Grace Blackwell (SM 121, aarch64)
# Output: faiss-gpu-${FAISS_CUDA_TAG} wheel (manylinux aarch64 platform tag)
#         + libfaiss-aarch64-${FAISS_CUDA_TAG}.so / libfaiss_c-aarch64-${FAISS_CUDA_TAG}.so
# cuVS:   uses libcuvs-spark.so from github.com/zbrad/cuvs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAISS_ROOT="${FAISS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
# CUDA version (single source of truth — bump in cuda_env.sh for cu133)
source "$SCRIPT_DIR/cuda_env.sh"
BUILD_OUTPUT_DIR="${FAISS_ROOT}/build_output_aarch64"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found. Set CUDA_HOME or install CUDA $FAISS_CUDA_VER."
        exit 1
    fi
    log_info "CUDA: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"

    CUVS_REPO="${CUVS_REPO:-/home/zbrad/gh/cuvs}"
    CUVS_DIR="${CUVS_DIR:-${CUVS_REPO}/cpp/build}"
    if [[ ! -f "${CUVS_DIR}/libcuvs-spark.so" ]]; then
        log_error "libcuvs-spark.so not found at ${CUVS_DIR}"
        log_error "Build it first: cd ${CUVS_REPO} && ./build_dgx_spark.sh"
        exit 1
    fi
    log_info "cuVS-spark: ${CUVS_DIR}/libcuvs-spark.so"

    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        log_error "Python not found."
        exit 1
    fi
    log_info "Python: $(${PYTHON:-python3} --version 2>&1)"

    for tool in cmake make swig; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool not found."
            exit 1
        fi
    done

    log_info "All prerequisites satisfied."
}

show_config() {
    log_info "DGX Spark Build Configuration:"
    echo "  Target GPU  : GB10 Grace Blackwell (SM 121)"
    echo "  Host arch   : aarch64 (sbsa-linux)"
    echo "  CUDA_HOME   : ${CUDA_HOME:-/usr/local/cuda}"
    echo "  CUDA_ARCHS  : 121-real"
    echo "  CUVS_DIR    : ${CUVS_DIR:-${CUVS_REPO:-/home/zbrad/gh/cuvs}/cpp/build}"
    echo "  Python      : $(${PYTHON:-python3} --version 2>&1)"
    echo "  C++ libs out: _libfaiss_stage_aarch64/"
    echo "  Wheel out   : $BUILD_OUTPUT_DIR/"
    echo "  Wheel name  : faiss-gpu-${FAISS_CUDA_TAG}$(CUDA_ARCHS=121 faiss_sm_suffix)"
}

build_lib() {
    log_info "Building C++ libraries (libfaiss-aarch64-${FAISS_CUDA_TAG}, libfaiss_c-aarch64-${FAISS_CUDA_TAG})..."
    bash "${SCRIPT_DIR}/build_lib_aarch64.sh"
}

build_pkg() {
    log_info "Building Python SWIG bindings..."
    bash "${SCRIPT_DIR}/build_pkg_aarch64.sh"
}

package_wheel() {
    log_info "Packaging wheel..."
    bash "${SCRIPT_DIR}/package_wheel_aarch64.sh"
}

cleanup() {
    log_warn "Cleaning up Spark build artifacts..."
    rm -rf "${FAISS_ROOT}/_build_aarch64" \
           "${FAISS_ROOT}/_build_python_aarch64"* \
           "${FAISS_ROOT}/_libfaiss_stage_aarch64"
    log_info "Cleanup complete."
}

main() {
    log_info "FAISS DGX Spark wheel build"

    check_prerequisites
    show_config

    mkdir -p "$BUILD_OUTPUT_DIR"

    case "${1:-all}" in
        lib)
            build_lib
            ;;
        pkg)
            build_lib
            build_pkg
            ;;
        wheel|all)
            build_lib
            build_pkg
            package_wheel
            ;;
        clean)
            cleanup
            ;;
        check)
            # already ran check_prerequisites + show_config above
            ;;
        *)
            echo "Usage: $0 [lib|pkg|wheel|all|clean|check]"
            echo "  lib   - Build C++ libraries only (libfaiss-aarch64-${FAISS_CUDA_TAG}.so)"
            echo "  pkg   - Build library + SWIG Python bindings"
            echo "  wheel - Build everything and package wheel (default)"
            echo "  all   - Same as 'wheel'"
            echo "  clean - Remove Spark build artifacts"
            echo "  check - Check prerequisites only"
            exit 1
            ;;
    esac

    if [[ "${1:-all}" != "check" && "${1:-all}" != "clean" ]]; then
        log_info "Build complete!"
        log_info "C++ libraries staged in: _libfaiss_stage_aarch64/lib/"
        if ls "$BUILD_OUTPUT_DIR"/faiss_gpu_${FAISS_CUDA_TAG}*.whl 2>/dev/null | grep -q .; then
            log_info "Wheel files:"
            ls -lh "$BUILD_OUTPUT_DIR"/faiss_gpu_${FAISS_CUDA_TAG}*.whl \
                   "$BUILD_OUTPUT_DIR"/repaired/faiss_gpu_${FAISS_CUDA_TAG}*.whl 2>/dev/null || true
        fi
    fi
}

main "$@"
