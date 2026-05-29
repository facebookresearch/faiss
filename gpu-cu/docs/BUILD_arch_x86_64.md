# Building FAISS-GPU Wheel for CUDA 13.2 and Python 3.14

This guide walks through building a FAISS-GPU wheel for CUDA 13.2 targeting Python 3.14.

## Quick start (WSL on Windows)

The fastest path on Windows is WSL 2 (Ubuntu). One-time Intel oneAPI MKL install
inside WSL:

```bash
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/intel-sw-products.gpg
echo "deb [signed-by=/usr/share/keyrings/intel-sw-products.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install -y intel-oneapi-mkl-devel
```

Then build and verify (from PowerShell or inside WSL):

```powershell
wsl -e bash gpu-cu/wsl/build.sh             # full build → build_output/ (log: /tmp/faiss_build.log)
wsl -e bash gpu-cu/wsl/verify.sh --install  # install wheel + CPU/GPU sanity check
```

`gpu-cu/wsl/env.sh` sets the WSL build environment and is sourced by the above.
Override the CUDA version or GPU archs per invocation:

```bash
FAISS_CUDA_VER=13.3 source gpu-cu/wsl/env.sh                  # build against CUDA 13.3
wsl -e bash -c "CUDA_ARCHS='89' bash gpu-cu/wsl/build.sh"     # RTX 4090 only → faiss-gpu-cu132-sm89
```

The rest of this guide covers the general (non-WSL) build, options, and the
aarch64 / DGX Spark pipeline.

## Prerequisites

- CUDA 13.2 toolkit installed
- Python 3.14 
- Build tools: CMake (>=3.24.0), SWIG (4.0), make, C++20 compiler
- Dependencies: Intel MKL, numpy, setuptools
- 8GB+ free disk space for build

## Quick Setup

### Option 1: Using Conda (Recommended)

```bash
# Create a conda environment with required dependencies
conda create -n faiss-gpu-cu132-py314 \
  -c pytorch \
  -c nvidia \
  -c conda-forge \
  python=3.14 \
  cmake>=3.24.0 \
  swig=4.0 \
  make=4.2 \
  cuda-toolkit=13.2 \
  mkl-devel>=2024.2.2 \
  gcc=12.4 \
  numpy \
  setuptools

conda activate faiss-gpu-cu132-py314
```

### Option 2: System Installation

Ensure the following are installed on your system:
- CUDA 13.2: `/usr/local/cuda-13.2` (or set `CUDA_HOME` variable)
- Python 3.14 with development headers
- CMake >= 3.24.0
- Intel MKL development libraries
- GCC 12.4+ with C++20 support

## Build Instructions

### 1. Set Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-13.2  # Adjust if using different path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_ARCHS="75;80;86;89;90;120"  # Adjust for your GPU architectures
```

### 1.1 Set Intel MKL Paths (Required)

`gpu-cu/scripts/build_lib_x86_64.sh` requires Intel MKL (`mkl_rt`) and may need explicit paths when auto-detection does not match your shell environment.

**WSL/Linux bash accessing Windows oneAPI install:**
```bash
export MKL_ROOT="/mnt/c/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
export MKL_LIB="$MKL_ROOT/lib/mkl_rt.lib"
export MKL_INCLUDE_DIR="$MKL_ROOT/include"
ls -l "$MKL_LIB" "$MKL_INCLUDE_DIR/mkl.h"
```

**Git Bash on Windows (MSYS path style):**
```bash
export MKL_ROOT="/c/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
export MKL_LIB="$MKL_ROOT/lib/mkl_rt.lib"
export MKL_INCLUDE_DIR="$MKL_ROOT/include"
ls -l "$MKL_LIB" "$MKL_INCLUDE_DIR/mkl.h"
```

**Find your exact installed MKL runtime from CMD:**
```cmd
where /r "C:\Program Files (x86)\Intel\oneAPI\mkl" mkl_rt*
```

Use the versioned MKL directory (for example `.../mkl/2025.3`) if `latest` symlink/path resolution behaves differently between shells.

Available CUDA architectures (CUDA 13.2 supported):
- `75`: Turing (RTX 2080, RTX 2060)
- `80`: Ampere (A100, RTX 3090)
- `86`: Ampere (RTX 3080 Ti, RTX 3070)
- `89`: Ada Lovelace (RTX 4090, RTX 4080)
- `90`: Hopper (H100)
- `120`: Blackwell (GB200, B200, RTX 5090+)
- `121`: Blackwell (GB10 Grace Blackwell / DGX Spark, aarch64)

> **Note:** Volta (70) and Turing (75) are **not supported** in CUDA 13.2 — NVIDIA removed
> offline compilation and library support in CUDA 13.0. For older GPUs, use CUDA 12.x.
> See: [CUDA 13.2 Release Notes - Deprecated Architectures](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#deprecated-architectures)

### 2. Build the Wheel

Use one of the provided build scripts:

**Automated build (recommended):**
```bash
bash gpu-cu/scripts/build_wheel_x86_64.sh
```

**Manual build:**
```bash
# Step 1: Build the C++ library
bash gpu-cu/scripts/build_lib_x86_64.sh

# Step 2: Build Python bindings and wheel
bash gpu-cu/scripts/build_pkg_x86_64.sh
```

### 3. Find the Built Wheel

The wheel will be located in `build_output/` directory:
```bash
ls -lh build_output/faiss_gpu*.whl
```

## Installation

To install the built wheel:

```bash
pip install build_output/faiss_gpu-*.whl
```

Verify installation:
```bash
python -c "import faiss; print(faiss.__version__); print(faiss.gpuGetNumDevices())"
```

## Customization Options

During build, you can customize:

- **GPU Architectures**: Set `CUDA_ARCHS` environment variable (space/semicolon separated)
  - Examples:
    - `CUDA_ARCHS="120"` - RTX 5090 (Blackwell) only
    - `CUDA_ARCHS="121-real"` - DGX Spark GB10 (SM 121, aarch64) only — see DGX Spark build below
    - `CUDA_ARCHS="90"` - H100 (Hopper) only
    - `CUDA_ARCHS="90;120"` - Hopper + Blackwell
    - `CUDA_ARCHS="75;80;86;89;90;120"` - All supported (default)

## DGX Spark / aarch64 build

This guide covers x86_64. For the ARM build (NVIDIA DGX Spark, GB10 Grace
Blackwell, SM 121 — OpenBLAS + cuVS instead of MKL/AVX), see the dedicated
**[BUILD_arch_aarch64.md](BUILD_arch_aarch64.md)** (`make build-aarch64`).

## More options

- **Optimization Level**: 
  - `generic`: Baseline optimization
  - `avx2`: AVX2 SIMD optimizations (default)
  - `avx512`: AVX512 optimizations
  - `avx512_spr`: Intel Sapphire Rapids optimizations

- **Build Type**: Add `-DCMAKE_BUILD_TYPE=Debug` for debug symbols

## Troubleshooting

**"CUDA not found"**
- Ensure CUDA 13.2 is installed and `CUDA_HOME` is set correctly
- Check: `ls $CUDA_HOME/bin/nvcc`

**"nvcc architecture mismatch"**
- List available architectures: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader --format=csv`
- Convert to CUDA_ARCHS format (e.g., 8.0 → 80)

**"Python development headers not found"**
- Install: `sudo apt install python3.14-dev` (or equivalent for your system)
- Or use conda environment with python-dev package

**"Build runs out of memory"**
- Reduce parallel jobs: `make -j4` instead of `-j$(nproc)`
- Edit build scripts and change `$(nproc)` to desired number

**"swig: command not found"**
- Install: `conda install swig=4.0` or `sudo apt install swig`

**"MKL runtime library not found"**
- Verify the shell path style matches your environment:
  - WSL/Linux bash: `/mnt/c/...`
  - Git Bash: `/c/...`
- Confirm files exist: `mkl_rt.lib` and `include/mkl.h`
- Use a versioned MKL path (e.g. `.../mkl/2025.3`) instead of `latest` if needed

## Performance Notes

- First build takes 10-30 minutes depending on GPU count and machine
- Subsequent builds use CMake cache for faster incremental builds
- Wheel size: ~300-500MB (includes GPU kernels)

## Testing the Wheel

```bash
# Run FAISS tests
python -c "from faiss import gpu; gpu.StandardGpuResources()"

# Run benchmarks
cd benchs/
python bench_*.py
```

## Cleaning Up

```bash
# Remove build files but keep wheel
bash gpu-cu/scripts/clean_build.sh

# Remove everything including wheels
rm -rf build/ _build* _libfaiss_stage/ build_output/
```
