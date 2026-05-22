# FAISS-GPU CUDA 13.2 Wheel Builder

Quick start guide for building the `faiss-gpu-cu132` wheel on **Windows via WSL 2** with CUDA 13.2 and Intel MKL.

## Prerequisites

| Requirement | Where to get it |
|---|---|
| WSL 2 (Ubuntu 22.04 / 24.04) | `wsl --install` in PowerShell |
| CUDA 13.2 Toolkit (inside WSL) | [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) |
| Intel oneAPI MKL (inside WSL) | `apt install intel-oneapi-mkl-devel` after adding Intel apt repo — see below |
| cmake ≥ 3.24, swig, make | `sudo apt install cmake swig make` |
| Python 3.10–3.12 + numpy | `sudo apt install python3 python3-dev python3-numpy` |

**Intel MKL apt install (one-time, inside WSL):**
```bash
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/intel-sw-products.gpg
echo "deb [signed-by=/usr/share/keyrings/intel-sw-products.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install -y intel-oneapi-mkl-devel
# Verify:
ls /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
```

## ⚡ Quick Start

### 1. Source the build environment

```bash
# Inside WSL, from the repo root:
source gpu-cu132/wsl/env.sh
```

Or set overrides before sourcing:
```bash
CUDA_ARCHS="89;90" source gpu-cu132/wsl/env.sh   # Ada + Hopper only
```

### 2. Build the wheel

```powershell
# From PowerShell (recommended):
wsl -e bash gpu-cu132/wsl/build.sh
```

```bash
# Or from inside WSL:
bash gpu-cu132/wsl/build.sh
```

Build log is tee'd to `/tmp/faiss_build.log`.

### 3. Verify the install

```powershell
# Install wheel + run CPU/GPU checks:
wsl -e bash gpu-cu132/wsl/verify.sh --install

# Re-run checks without reinstalling:
wsl -e bash gpu-cu132/wsl/verify.sh
```

Expected output:
```
  faiss version : 1.14.1
  GPU count     : 1
  CPU search    : OK (1000 vectors, top-4)
  GPU search    : OK (GPU 0)
✓ All checks passed
```

## 📋 Build Scripts

### WSL helper scripts (`gpu-cu132/wsl/`)

| Script | Purpose | Usage |
|---|---|---|
| `gpu-cu132/wsl/env.sh` | Set CUDA/MKL/FAISS env vars | `source gpu-cu132/wsl/env.sh` |
| `gpu-cu132/wsl/build.sh` | Full build: lib → SWIG → wheel | `wsl -e bash gpu-cu132/wsl/build.sh` |
| `gpu-cu132/wsl/verify.sh` | Import + CPU + GPU sanity check | `wsl -e bash gpu-cu132/wsl/verify.sh` |

### Core build scripts (called by the above)

| Script | Purpose |
|---|---|
| `gpu-cu132/build_lib_cuda132.sh` | Build C++ library (`libfaiss*.so`) via CMake |
| `gpu-cu132/build_pkg_cuda132.sh` | Build Python SWIG bindings (`_swigfaiss*.so`) |
| `gpu-cu132/package_wheel.sh` | Create `.whl` from built artifacts |
| `gpu-cu132/clean_build.sh` | Remove all build directories and artifacts |
| `Makefile` | Orchestrates all stages — `make build` runs everything |

## 🔧 Common Commands

```powershell
# Full build (all 7 architectures)
wsl -e bash gpu-cu132/wsl/build.sh

# Build for your GPU arch only (faster)
wsl -e bash -c "CUDA_ARCHS='89' bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"   # RTX 4090 (Ada)
wsl -e bash -c "CUDA_ARCHS='90' bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"   # H100 (Hopper)
wsl -e bash -c "CUDA_ARCHS='86' bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"   # RTX 3080 Ti (Ampere)
wsl -e bash -c "CUDA_ARCHS='90;120' bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"  # Hopper + Blackwell

# Parallel jobs
wsl -e bash -c "FAISS_BUILD_JOBS=16 bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"

# Clean and rebuild
wsl -e bash -c "bash /mnt/f/GitHub/faiss/gpu-cu132/clean_build.sh && bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"

# Install wheel (use --break-system-packages on Ubuntu 24.04)
wsl -e bash -c "pip3 install /mnt/f/GitHub/faiss/build_output/faiss_gpu_cu132-*.whl --break-system-packages"

# Verify
wsl -e bash gpu-cu132/wsl/verify.sh
```

## 📊 GPU Architecture Codes

Set `CUDA_ARCHS` environment variable before building:

| CUDA Code | SM Code | GPU Examples |
|-----------|---------|------------------|
| `75` | sm_75 | RTX 2080, RTX 2060 (Turing) |
| `80` | sm_80 | A100, RTX 3090 (Ampere) |
| `86` | sm_86 | RTX 3080 Ti, RTX 3070 (Ampere) |
| `89` | sm_89 | RTX 4090, RTX 4080 (Ada) |
| `90` | sm_90 | H100 (Hopper) |
| `120` | sm_120 | GB200, B200, RTX 5090+ (Blackwell) |
| `121` | sm_121 | GB10 Grace Blackwell — DGX Spark (aarch64 only) |

```powershell
# Build for multiple architectures
wsl -e bash -c "CUDA_ARCHS='75;80;86;89;90;120' bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"
```

## 📁 Directory Structure After Build

```
faiss-gpu-cu132/
├── build_output/         # Output wheels
│   └── faiss_gpu_cu132-*.whl
├── _build/               # C++ build artifacts
├── _build_python_*/      # Python build artifacts
├── _libfaiss_stage/      # Staged libraries
└── [source files]
```

## 🐛 Troubleshooting

**CUDA not found:**
```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=$CUDA_HOME/bin:$PATH
```

**Python dev headers missing:**
```bash
# WSL / Ubuntu
sudo apt install python3-dev
```

**"make: parallel limits exceeded" / memory issues during build:**
```powershell
wsl -e bash -c "FAISS_BUILD_JOBS=4 bash /mnt/f/GitHub/faiss/gpu-cu132/wsl/build.sh"
```

**"MKL runtime library not found":**
- Confirm Intel MKL is installed inside WSL: `ls /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`
- If installed to a non-default path, override before building:
  ```bash
  export MKL_ROOT=/path/to/mkl
  source gpu-cu132/wsl/env.sh
  ```
- Re-install: `sudo apt install --reinstall intel-oneapi-mkl-devel`

## 📖 Full Documentation

See [BUILD_WHEEL_CUDA132.md](BUILD_WHEEL_CUDA132.md) for complete documentation.

## 📝 Notes

- First build takes 10-30 minutes
- Uses AVX2 optimization by default
- Wheel size: ~300-500MB
- Requires 8GB+ free disk space

## ✅ Testing

```powershell
# Quick sanity check (CPU + GPU)
wsl -e bash gpu-cu132/wsl/verify.sh

# Full test suite (inside WSL)
wsl -e bash -c "cd /mnt/f/GitHub/faiss && python3 -m pytest tests/ -x -q"

# GPU-specific tests
wsl -e bash -c "cd /mnt/f/GitHub/faiss/faiss/gpu/test && python3 -m pytest test_*.py -x -q"
```

## 📦 Wheel Details

The built wheel includes:
- FAISS Python bindings (faiss.py module)
- GPU index implementations
- CUDA kernels for supported architectures
- C API bindings
- Optimized SIMD variants (AVX2, AVX512)

## 🔗 Resources

- **Official Repo:** https://github.com/facebookresearch/faiss
- **Documentation:** https://github.com/facebookresearch/faiss/wiki
- **Issue Tracker:** https://github.com/facebookresearch/faiss/issues
- **Build Guide:** [BUILD_WHEEL_CUDA132.md](BUILD_WHEEL_CUDA132.md)
