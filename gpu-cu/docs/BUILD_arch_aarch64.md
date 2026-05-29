# Building FAISS-GPU Wheel for aarch64 / NVIDIA DGX Spark

This guide builds a FAISS-GPU wheel for **aarch64** (ARM), targeting the NVIDIA
DGX Spark — GB10 Grace Blackwell Superchip, compute capability **SM 121**.

Unlike the x86_64 build, ARM has **no Intel MKL**: this pipeline uses **OpenBLAS**
for CPU BLAS and **NVIDIA cuVS** (the `libcuvs-spark` build from
[zbrad/cuvs](https://github.com/zbrad/cuvs)) for GPU acceleration. See
[BUILD_arch_x86_64.md](BUILD_arch_x86_64.md) for the x86_64 build.

## Prerequisites

- CUDA 13.2 toolkit for **aarch64 / sbsa-linux** (`/usr/local/cuda-13.2`, or set `CUDA_HOME`)
- Python 3 with development headers, numpy, setuptools
- Build tools: CMake (>=3.24.0), SWIG (4.0), make, a C++20 compiler
- **OpenBLAS**: `sudo apt install libopenblas-dev` (default `/usr/lib/aarch64-linux-gnu/libopenblas.so`)
- **libcuvs-spark** built for SM 121 (see next section)
- 8GB+ free disk space

> Intel MKL is **not** used on ARM. If you set `MKL_*` variables they are ignored
> by this pipeline.

## cuVS-spark dependency

The aarch64 build links the SM 121 cuVS library `libcuvs-spark.so`, built from the
[zbrad/cuvs](https://github.com/zbrad/cuvs) fork. Build it first:

```bash
git clone https://github.com/zbrad/cuvs /home/zbrad/gh/cuvs
cd /home/zbrad/gh/cuvs && ./build_dgx_spark.sh
# produces cpp/build/libcuvs-spark.so
```

The build scripts look for it at `${CUVS_DIR}/libcuvs-spark.so`, where
`CUVS_DIR` defaults to `${CUVS_REPO}/cpp/build` and `CUVS_REPO` defaults to
`/home/zbrad/gh/cuvs`. Override either if your checkout differs:

```bash
export CUVS_REPO=/path/to/cuvs          # or
export CUVS_DIR=/path/to/cuvs/cpp/build
```

## Quick start

```bash
# Full build: C++ libs → SWIG bindings → wheel
make build-aarch64
# or directly:
bash gpu-cu/scripts/build_wheel_aarch64.sh
```

Output wheel lands in `build_output_aarch64/` (and `build_output_aarch64/repaired/`
after `auditwheel`). Install and check:

```bash
pip install build_output_aarch64/repaired/faiss_gpu_cu132_sm121-*.whl
python -c "import faiss; print(faiss.__version__, faiss.get_num_gpus())"
```

## Build steps (manual)

```bash
bash gpu-cu/scripts/build_lib_aarch64.sh      # libfaiss-aarch64-cu132.so + libfaiss_c-aarch64-cu132.so
bash gpu-cu/scripts/build_pkg_aarch64.sh      # SWIG Python bindings (generic opt-level, no AVX)
bash gpu-cu/scripts/package_wheel_aarch64.sh  # wheel + auditwheel repair
```

| Script | Purpose |
|--------|---------|
| `build_lib_aarch64.sh` | C++ library — SM 121, OpenBLAS, links `libcuvs-spark.so` |
| `build_pkg_aarch64.sh` | SWIG bindings (generic opt-level; ARM has no AVX) |
| `package_wheel_aarch64.sh` | Wheel packaging + `auditwheel repair` (manylinux aarch64) |
| `build_wheel_aarch64.sh` | Unified orchestrator (`all` / `lib` / `pkg` / `wheel` / `clean` / `check`) |

## CUDA version selection

The CUDA version is a single input, shared with the x86_64 pipeline via
`gpu-cu/scripts/cuda_env.sh`. Specify it per build (the tag is derived):

```bash
make build-aarch64 FAISS_CUDA_VER=13.3        # → faiss-gpu-cu133-sm121
FAISS_CUDA_TAG=cu133 bash gpu-cu/scripts/build_wheel_aarch64.sh
```

On a host with multiple toolkits, `CUDA_HOME` auto-resolves to
`/usr/local/cuda-<ver>` (override `CUDA_HOME` to force a path). See
[WHEEL_NAMING.md](WHEEL_NAMING.md) for the full version/naming scheme.

## Outputs

| Artifact | Location | Notes |
|----------|----------|-------|
| `libfaiss-aarch64-cu132.so` | `_libfaiss_stage_aarch64/lib/` | Main FAISS GPU library, SM 121 (`FAISS_OUTPUT_NAME`) |
| `libfaiss_c-aarch64-cu132.so` | `_libfaiss_stage_aarch64/lib/` | C API wrapper (`FAISS_C_OUTPUT_NAME`) |
| `faiss-gpu-cu132-sm121` wheel | `build_output_aarch64/` | Single-arch wheel; `manylinux_*_aarch64` platform tag |
| links `libcuvs-spark.so` | `zbrad/cuvs` build | SM 121 native cuVS |

The `cu132`/`sm121` portions track `FAISS_CUDA_TAG` and `CUDA_ARCHS`. Because the
DGX Spark build always targets a single GPU arch (SM 121), the package name
carries the `-sm121` suffix; see [WHEEL_NAMING.md](WHEEL_NAMING.md).

## Troubleshooting

**"libcuvs-spark.so not found"**
- Build it: `cd $CUVS_REPO && ./build_dgx_spark.sh`, or set `CUVS_DIR` to its location.

**"OpenBLAS not found"**
- Install: `sudo apt install libopenblas-dev`
- Non-default path: `export OPENBLAS_LIB=/path/to/libopenblas.so`

**"CUDA not found" / version mismatch**
- Ensure the aarch64 CUDA toolkit is installed; `ls $CUDA_HOME/bin/nvcc`
- If `nvcc` reports a different version than `FAISS_CUDA_VER`, set `CUDA_HOME` to the matching toolkit.

**"Python development headers not found"**
- `sudo apt install python3-dev`

**Build runs out of memory**
- Reduce parallelism: `FAISS_BUILD_JOBS=4 make build-aarch64`

## Cleaning up

```bash
bash gpu-cu/scripts/build_wheel_aarch64.sh clean   # remove aarch64 build dirs
# or, both arches:
bash gpu-cu/scripts/clean_build.sh
```
