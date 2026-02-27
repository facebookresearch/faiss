#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Automated script to build and install Faiss from source using uv

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENABLE_GPU="OFF"
PYTHON_EXECUTABLE=""
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
GPU_DETECTED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --gpu)
            ENABLE_GPU="ON"
            GPU_DETECTED=true
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --python PATH     Path to Python executable (default: auto-detect)"
            echo "  --gpu             Enable GPU support (default: auto-detect)"
            echo "  --jobs N          Number of parallel jobs (default: auto)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect GPU type and enable if available
ENABLE_ROCM="OFF"
if [ "$GPU_DETECTED" = false ]; then
    echo -e "${YELLOW}Auto-detecting GPU availability...${NC}"
    
    # Check for CUDA (NVIDIA)
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ] || [ -d "/opt/cuda" ]; then
        echo -e "${GREEN}GPU automatically detected: CUDA - enabling GPU support${NC}"
        ENABLE_GPU="ON"
    # Check for ROCm (AMD)
    elif [ -d "/opt/rocm" ] || command -v rocm-smi &> /dev/null; then
        echo -e "${GREEN}GPU automatically detected: ROCm - enabling GPU support${NC}"
        ENABLE_GPU="ON"
        ENABLE_ROCM="ON"
    else
        echo -e "${GREEN}No GPU detected - using CPU-only build${NC}"
        ENABLE_GPU="OFF"
    fi
    echo ""
elif [ "$ENABLE_GPU" = "ON" ]; then
    # GPU was explicitly enabled, check if it's ROCm
    echo -e "${YELLOW}Checking GPU type...${NC}"
    if [ -d "/opt/rocm" ] || command -v rocm-smi &> /dev/null; then
        echo -e "${GREEN}ROCm detected - enabling ROCm support${NC}"
        ENABLE_ROCM="ON"
    else
        echo -e "${GREEN}Using CUDA (default)${NC}"
    fi
    echo ""
fi

echo -e "${GREEN}=== Faiss Installation Script with uv ===${NC}"
echo ""

# Check for required tools
echo -e "${YELLOW}Checking dependencies...${NC}"
MISSING_DEPS=()

if ! command -v cmake &> /dev/null; then
    MISSING_DEPS+=("cmake")
fi

if ! command -v make &> /dev/null; then
    MISSING_DEPS+=("make")
fi

# Check for Python dependencies (always needed for this script)
if ! command -v swig &> /dev/null; then
    MISSING_DEPS+=("swig")
fi

if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add common uv installation paths to PATH
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    
    # Try to source the shell profile if uv was just installed
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env" 2>/dev/null || true
    fi
    
    if ! command -v uv &> /dev/null; then
        MISSING_DEPS+=("uv")
    fi
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo -e "${RED}Missing required dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "Please install them and try again."
    exit 1
fi

echo -e "${GREEN}All dependencies found.${NC}"
echo ""

# Detect Python if not specified
if [ -z "$PYTHON_EXECUTABLE" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_EXECUTABLE=$(which python3)
    elif command -v python &> /dev/null; then
        PYTHON_EXECUTABLE=$(which python)
    else
        echo -e "${RED}Python not found. Please specify with --python PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}Using Python: $PYTHON_EXECUTABLE${NC}"
fi

# If we're in a virtual environment, ensure numpy is installed for CMake
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Installing numpy in virtual environment (required for CMake configuration)...${NC}"
    uv pip install numpy || python3 -m pip install numpy --quiet
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configure OpenMP flags based on OS
# CMake should auto-detect OpenMP on Linux, but macOS needs special handling
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: Check for libomp in common Homebrew locations
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        # Apple Silicon Homebrew
        export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
        export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
        OPENMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        OPENMP_CXX_LIB_NAMES="omp"
        OPENMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"
        OPENMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        OPENMP_C_LIB_NAMES="omp"
    elif [ -d "/usr/local/opt/libomp" ]; then
        # Intel Homebrew
        export LDFLAGS="-L/usr/local/opt/libomp/lib"
        export CPPFLAGS="-I/usr/local/opt/libomp/include"
        OPENMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
        OPENMP_CXX_LIB_NAMES="omp"
        OPENMP_omp_LIBRARY="/usr/local/opt/libomp/lib/libomp.dylib"
        OPENMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
        OPENMP_C_LIB_NAMES="omp"
    else
        echo -e "${YELLOW}Warning: libomp not found. OpenMP support may not work.${NC}"
        echo "Install with: brew install libomp"
        echo "CMake will try to find OpenMP automatically, but it may fail."
    fi
else
    # Linux and other Unix-like systems: CMake should auto-detect OpenMP
    # Most Linux distributions have OpenMP in standard locations
    echo -e "${GREEN}Using system OpenMP (CMake will auto-detect)${NC}"
fi

# Step 1: Configure with CMake
echo -e "${YELLOW}Step 1: Configuring build with CMake...${NC}"
CMAKE_ARGS=(
    -B build
    -DCMAKE_BUILD_TYPE=Release
    -DFAISS_ENABLE_PYTHON=ON
    -DFAISS_ENABLE_GPU="$ENABLE_GPU"
    -DFAISS_ENABLE_ROCM="$ENABLE_ROCM"
    -DBUILD_TESTING=OFF
    -DPython_EXECUTABLE="$PYTHON_EXECUTABLE"
)

if [ -n "$OPENMP_CXX_FLAGS" ]; then
    CMAKE_ARGS+=(
        -DOpenMP_CXX_FLAGS="$OPENMP_CXX_FLAGS"
        -DOpenMP_CXX_LIB_NAMES="$OPENMP_CXX_LIB_NAMES"
        -DOpenMP_omp_LIBRARY="$OPENMP_omp_LIBRARY"
        -DOpenMP_C_FLAGS="$OPENMP_C_FLAGS"
        -DOpenMP_C_LIB_NAMES="$OPENMP_C_LIB_NAMES"
    )
fi

cmake "${CMAKE_ARGS[@]}" .

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration complete.${NC}"
echo ""

# Step 2: Build C++ library
echo -e "${YELLOW}Step 2: Building C++ library...${NC}"
make -C build -j"$NUM_JOBS" faiss

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}C++ library built successfully.${NC}"
echo ""

# Step 3: Build Python bindings
echo -e "${YELLOW}Step 3: Building Python bindings...${NC}"
make -C build -j"$NUM_JOBS" swigfaiss

if [ $? -ne 0 ]; then
    echo -e "${RED}Python bindings build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Python bindings built successfully.${NC}"
echo ""

# Step 4: Install with uv
echo -e "${YELLOW}Step 4: Installing Python package with uv...${NC}"
cd build/faiss/python

# uv works best when run from within a virtual environment
# If we're in a venv, use it directly. Otherwise, try to use the specified Python
if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
    # Already in a virtual environment - uv will use it automatically
    echo -e "${GREEN}Using active virtual environment: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}${NC}"
    uv pip install .
elif [ -n "$PYTHON_EXECUTABLE" ]; then
    # Check if the Python executable is in a virtual environment
    PYTHON_DIR=$(dirname "$PYTHON_EXECUTABLE")
    if [ -f "$PYTHON_DIR/activate" ] || [ -d "$PYTHON_DIR/../lib/python"* ] || [ -d "$PYTHON_DIR/../Lib" ]; then
        # It's a venv - activate it first
        echo -e "${GREEN}Detected virtual environment, activating...${NC}"
        source "$PYTHON_DIR/activate" 2>/dev/null || true
        uv pip install .
    else
        # System Python - try to install, but warn about permissions
        echo -e "${YELLOW}Note: Installing to system Python. If you get permission errors, consider using a virtual environment.${NC}"
        echo -e "${YELLOW}You can create one with: python3 -m venv venv && source venv/bin/activate${NC}"
        uv pip install . --python "$PYTHON_EXECUTABLE" || {
            echo -e "${RED}Installation failed!${NC}"
            echo -e "${YELLOW}Tip: Create a virtual environment first:${NC}"
            echo -e "  python3 -m venv venv && source venv/bin/activate"
            echo -e "  Then run this script again"
            exit 1
        }
    fi
else
    # No Python specified and not in venv - try default, but warn
    echo -e "${YELLOW}Warning: Not in a virtual environment. Installation may require sudo or fail due to permissions.${NC}"
    echo -e "${YELLOW}Consider creating a virtual environment first: python3 -m venv venv && source venv/bin/activate${NC}"
    uv pip install . || {
        echo -e "${RED}Installation failed!${NC}"
        echo -e "${YELLOW}Tip: Create a virtual environment first:${NC}"
        echo -e "  python3 -m venv venv && source venv/bin/activate"
        echo -e "  Or use uv: uv venv && source .venv/bin/activate"
        echo -e "  Then run this script again"
        exit 1
    }
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Installation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Python package installed successfully.${NC}"
echo ""

cd "$SCRIPT_DIR"

echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo ""
echo "Faiss has been successfully built and installed."
echo "You can now import faiss in Python:"
echo "  python3 -c 'import faiss; print(faiss.__version__)'"

