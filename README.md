# Faiss

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Meta's [Fundamental AI Research](https://ai.facebook.com/) group.

> **Note:** This repository includes an **optional** CUDA 13.2 GPU wheel toolkit under [gpu-cu132](gpu-cu132). Standard FAISS usage is unchanged.

## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Introduction

Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Some of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. Other methods, like HNSW and NSG add an indexing structure on top of the raw vectors to make searching more efficient.

The GPU implementation can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace `IndexFlatL2` with `GpuIndexFlatL2`) and copies to/from GPU memory are handled automatically. Results will be faster however if both input and output remain resident on the GPU. Both single and multi-GPU usage is supported.

### Supported GPU Architectures for CUDA 13.2 Builds

**CUDA 13.2 Architecture Requirements:**
This build is optimized for Compute Capability 8.0 and higher (Ampere generation and newer). NVIDIA's CUDA 13.2 toolkit no longer provides optimized libraries for pre-Ampere architectures (Volta 7.0, Turing 7.5). This build focuses on modern, actively supported GPU generations.

**Supported Architectures:**
- **Architecture 75** - NVIDIA RTX 2080, RTX 2060 (Turing)
- **Architecture 80** - NVIDIA A100, RTX 3090 (Ampere)
- **Architecture 86** - NVIDIA RTX 3080 Ti, RTX 3070 (Ampere)
- **Architecture 89** - NVIDIA RTX 4090, RTX 4080 (Ada)
- **Architecture 90** - NVIDIA H100 (Hopper)
- **Architecture 100** - NVIDIA GB10 Grace Blackwell — **DGX Spark systems**
- **Architecture 120** - NVIDIA RTX 5090 (Blackwell)

All architectures are included by default. Customize with `CUDA_ARCHS` environment variable.

**Note:** Volta (70) and below (Maxwell 5.x, Pascal 6.x) are not supported in CUDA 13.2 — removed in CUDA 13.0. Turing (75) remains supported.

**References:**
- [CUDA 13.2 Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) - Official NVIDIA documentation
- [Deprecated Architectures in CUDA 13.0+](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#deprecated-architectures) - Details on Maxwell, Pascal, and Volta removal

## Installing

Faiss comes with precompiled libraries for Anaconda in Python, see [faiss-cpu](https://anaconda.org/pytorch/faiss-cpu), [faiss-gpu](https://anaconda.org/pytorch/faiss-gpu) and [faiss-gpu-cuvs](https://anaconda.org/pytorch/faiss-gpu-cuvs). The library is mostly implemented in C++, the only dependency is a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. Optional GPU support is provided via CUDA or AMD ROCm, and the Python interface is also optional. The backend GPU implementations of NVIDIA [cuVS](https://github.com/rapidsai/cuvs) can also be enabled optionally. It compiles with cmake. See [INSTALL.md](INSTALL.md) for details.

### Building FAISS-GPU for CUDA 13.2

This repository includes build scripts and tooling for creating FAISS-GPU wheels optimized for CUDA 13.2 with Python 3.14. Supports modern GPU architectures including H100 (Hopper), RTX 4090 (Ada), and RTX 5090 (Blackwell).

**Required dependency:** Intel MKL is required for `gpu-cu132/scripts/build_lib_cuda132.sh` (OpenBLAS is not used by this CUDA 13.2 build flow).

**Intel MKL source/install links:**
- [Intel oneAPI Base Toolkit Download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- [Intel oneAPI MKL Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

**Windows (PowerShell) install example:**
```powershell
winget install --id Intel.oneAPI.BaseToolkit -e --accept-source-agreements --accept-package-agreements
```

After installation, set `MKL_ROOT`, `MKL_LIB`, and `MKL_INCLUDE_DIR` if auto-detection does not find your installation.

**Quick Start (WSL 2 on Windows):**
```powershell
# Full build — C++ library + Python bindings + wheel
wsl -e bash gpu-cu132/wsl/build.sh

# Install wheel and run CPU + GPU verification
wsl -e bash gpu-cu132/wsl/verify.sh --install
```

See [QUICKSTART.md](gpu-cu132/docs/QUICKSTART.md) for prerequisites, per-architecture builds, and troubleshooting.

**Key Features:**
- ✅ Multi-GPU architecture support (75, 80, 86, 89, 90, 100, 120)
- ✅ Optimized builds with AVX2/AVX512 SIMD variants
- ✅ DGX Spark (GB10 Blackwell) optimizations included
- ✅ Comprehensive build documentation
- ✅ Environment verification tools
- ✅ Makefile targets for easy building

**Documentation:**
- [QUICKSTART.md](gpu-cu132/docs/QUICKSTART.md) - Build guide (WSL + scripts)
- [BUILD_WHEEL_CUDA132.md](gpu-cu132/docs/BUILD_WHEEL_CUDA132.md) - Complete build documentation
- [SETUP_COMPLETE.md](gpu-cu132/SETUP_COMPLETE.md) - Setup overview
- [BRANCH_CHANGES_SUMMARY.md](gpu-cu132/docs/BRANCH_CHANGES_SUMMARY.md) - Semantic branch change summary

**Build Scripts:**
- `gpu-cu132/wsl/build.sh` - WSL full build launcher (recommended)
- `gpu-cu132/wsl/verify.sh` - Install + CPU/GPU verification
- `gpu-cu132/wsl/env.sh` - Environment setup (source before manual steps)
- `gpu-cu132/scripts/build_lib_cuda132.sh` - Build C++ library
- `gpu-cu132/scripts/build_pkg_cuda132.sh` - Build Python package
- `gpu-cu132/scripts/package_wheel.sh` - Package as wheel
- `gpu-cu132/verify_environment.py` - Check prerequisites
- `Makefile` - Build targets and configuration

#### Build Process Overview

The build system uses a three-stage process:

1. **C++ Library Build** (`gpu-cu132/scripts/build_lib_cuda132.sh`)
   - Compiles FAISS C++ core with CMake
   - Generates optimized libraries for configured architectures
   - Produces `libfaiss.so`, `libfaiss_avx2.so`, `libfaiss_avx512.so`

2. **Python Bindings** (`gpu-cu132/scripts/build_pkg_cuda132.sh`)
   - Generates SWIG bindings for Python interface
   - Builds Python extension modules
   - Creates `swigfaiss` modules for different optimizations

3. **Wheel Packaging** (`gpu-cu132/scripts/package_wheel.sh`)
   - Packages Python module as `.whl` distribution
   - Output to `build_output/faiss_gpu-*.whl`

#### Common Build Commands

**For DGX Spark (GB10 Blackwell, aarch64):**
```bash
# DGX Spark uses a dedicated build pipeline (SM 121, aarch64)
bash gpu-cu132/scripts/build_wheel_spark.sh
```

**For RTX 4090:**
```bash
# Build for RTX 4090 only
CUDA_ARCHS="89" make build
```

**For RTX 5090:**
```bash
# Build for RTX 5090 (Blackwell)
CUDA_ARCHS="120" make build
```

**For Multiple Architectures:**
```bash
# Build for Hopper + Blackwell
CUDA_ARCHS="90;120" make build

# Build all supported (default)
make build
```

#### Advanced Customization

```bash
# Parallel build jobs
FAISS_BUILD_JOBS=8 make build

# Custom CUDA path
CUDA_HOME=/opt/cuda-13.2 make build

# Custom Python
PYTHON=python3.14 make build

# Clean build
make clean
```

## How Faiss works

Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to

- search time
- search quality
- memory used per index vector
- training time
- adding time
- need for external data for unsupervised training

The optional GPU implementation provides what is likely (as of March 2017) the fastest exact and approximate (compressed-domain) nearest neighbor search implementation for high-dimensional vectors, fastest Lloyd's k-means, and fastest small k-selection algorithm known. [The implementation is detailed here](https://arxiv.org/abs/1702.08734).

## Full documentation of Faiss

The following are entry points for documentation:

- the full documentation can be found on the [wiki page](http://github.com/facebookresearch/faiss/wiki), including a [tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started), a [FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ) and a [troubleshooting section](https://github.com/facebookresearch/faiss/wiki/Troubleshooting)
- the [doxygen documentation](https://faiss.ai/) gives per-class information extracted from code comments
- to reproduce results from our research papers, [Polysemous codes](https://arxiv.org/abs/1609.01882) and [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), refer to the [benchmarks README](benchs/README.md). For [
Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996), see the [link_and_code README](benchs/link_and_code)

## Authors

The main authors of Faiss are:
- [Hervé Jégou](https://github.com/jegou) initiated the Faiss project and wrote its first implementation
- [Matthijs Douze](https://github.com/mdouze) implemented most of the CPU Faiss
- [Jeff Johnson](https://github.com/wickedfoo) implemented all of the GPU Faiss
- [Lucas Hosseini](https://github.com/beauby) implemented the binary indexes and the build system
- [Chengqi Deng](https://github.com/KinglittleQ) implemented NSG, NNdescent and much of the additive quantization code.
- [Alexandr Guzhva](https://github.com/alexanderguzhva) many optimizations: SIMD, memory allocation and layout, fast decoding kernels for vector codecs, etc.
- [Gergely Szilvasy](https://github.com/algoriddle) build system, benchmarking framework.

## Reference

References to cite when you use Faiss in a research paper:
```
@article{douze2024faiss,
      title={The Faiss library},
      author={Matthijs Douze and Alexandr Guzhva and Chengqi Deng and Jeff Johnson and Gergely Szilvasy and Pierre-Emmanuel Mazaré and Maria Lomeli and Lucas Hosseini and Hervé Jégou},
      year={2024},
      eprint={2401.08281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
For the GPU version of Faiss, please cite:
```
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

## Join the Faiss community

For public discussion of Faiss or for questions, visit https://github.com/facebookresearch/faiss/discussions.

We monitor the [issues page](http://github.com/facebookresearch/faiss/issues) of the repository.
You can report bugs, ask questions, etc.

## Legal

Faiss is MIT-licensed, refer to the [LICENSE file](https://github.com/facebookresearch/faiss/blob/main/LICENSE) in the top level directory.

Copyright © Meta Platforms, Inc.

## Branch-Specific CUDA 13.2 Docs

For faiss-gpu-cu132 branch build instructions (CUDA 13.2 + MKL), see:

- gpu-cu132/docs/QUICKSTART.md
- gpu-cu132/docs/BUILD_WHEEL_CUDA132.md
- gpu-cu132/docs/SETUP_COMPLETE.md
