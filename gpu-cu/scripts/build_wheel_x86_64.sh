#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Unified build script for FAISS-GPU wheel — x86_64 (Intel MKL)
# Output: faiss-gpu-${FAISS_CUDA_TAG} wheel (manylinux x86_64 platform tag)
#         + libfaiss-x86_64-${FAISS_CUDA_TAG}.so / libfaiss_c-x86_64-${FAISS_CUDA_TAG}.so

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAISS_ROOT="${FAISS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
# CUDA version (single source of truth — bump in cuda_env.sh for cu133)
source "$SCRIPT_DIR/cuda_env.sh"
BUILD_OUTPUT_DIR="${FAISS_ROOT}/build_output"

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $@"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $@"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $@"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        log_error "CUDA not found. Please install CUDA $FAISS_CUDA_VER or set CUDA_HOME."
        exit 1
    fi
    
    cuda_version=$(nvcc --version | grep "release" | awk '{print $(NF)}')
    log_info "Found CUDA version: $cuda_version"
    
    # Check Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python not found"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 || python3 --version 2>&1)
    log_info "Found Python: $python_version"
    
    # Check required tools
    for tool in cmake make swig; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool not found. Please install it."
            exit 1
        fi
    done
    
    log_info "All prerequisites satisfied."
}

# Show configuration
show_config() {
    log_info "Build Configuration:"
    echo "  CUDA_HOME: ${CUDA_HOME:-/usr/local/cuda}"
    echo "  CUDA_ARCHS: ${CUDA_ARCHS:-75;80;86;89;90;120}"
    echo "  Python: $(python --version 2>&1 || python3 --version 2>&1)"
    echo "  Build output: $BUILD_OUTPUT_DIR"
}

# Build steps
build_lib() {
    log_info "Building C++ library (libfaiss)..."
    bash "${SCRIPT_DIR}/build_lib_x86_64.sh"
}

build_pkg() {
    log_info "Building Python package and wheel..."
    bash "${SCRIPT_DIR}/build_pkg_x86_64.sh"
}

package_wheel() {
    log_info "Packaging wheel..."
    bash "${SCRIPT_DIR}/package_wheel_x86_64.sh"
}

# Cleanup
cleanup() {
    log_warn "Cleaning up build artifacts..."
        rm -rf "${FAISS_ROOT}/_build" "${FAISS_ROOT}/_build_python"* "${FAISS_ROOT}/_libfaiss_stage" \
            "${FAISS_ROOT}/build" "${FAISS_ROOT}/faiss/python/build"
    log_info "Cleanup complete."
}

# Main build process
main() {
    log_info "Starting FAISS-GPU wheel build for CUDA $FAISS_CUDA_VER..."
    
    check_prerequisites
    show_config
    
    # Create output directory
    mkdir -p "$BUILD_OUTPUT_DIR"
    
    # Parse arguments
    case "${1:-all}" in
        lib)
            build_lib
            ;;
        pkg)
            build_lib
            build_pkg
            ;;
        wheel)
            build_lib
            build_pkg
            package_wheel
            ;;
        all)
            build_lib
            build_pkg
            package_wheel
            ;;
        clean)
            cleanup
            ;;
        check)
            check_prerequisites
            show_config
            ;;
        *)
            echo "Usage: $0 [lib|pkg|wheel|all|clean|check]"
            echo "  lib   - Build C++ library only"
            echo "  pkg   - Build library and Python package"
            echo "  wheel - Build everything and package wheel"
            echo "  all   - Same as 'wheel' (default)"
            echo "  clean - Remove build artifacts"
            echo "  check - Check prerequisites only"
            exit 1
            ;;
    esac
    
    if [[ "${1:-all}" != "check" ]]; then
        log_info "Build complete!"
        log_info "Wheel output directory: $BUILD_OUTPUT_DIR"
        if [[ -f "$BUILD_OUTPUT_DIR"/faiss_gpu*.whl ]]; then
            log_info "Wheel files:"
            ls -lh "$BUILD_OUTPUT_DIR"/faiss_gpu*.whl
        fi
    fi
}

main "$@"
