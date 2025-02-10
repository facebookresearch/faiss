# Installing Faiss via conda

The supported way to install Faiss is through [conda](https://docs.conda.io).
Stable releases are pushed regularly to the pytorch conda channel, as well as
pre-release nightly builds.

- The CPU-only faiss-cpu conda package is currently available on Linux (x86-64 and aarch64), OSX (arm64 only), and Windows (x86-64)
- faiss-gpu, containing both CPU and GPU indices, is available on Linux (x86-64 only) for CUDA 11.4 and 12.1
- faiss-gpu-cuvs [^1] package containing GPU indices provided by [NVIDIA cuVS](https://github.com/rapidsai/cuvs/) version 24.12, is available on Linux (x86-64 only) for CUDA 11.8 and 12.4.

To install the latest stable release:

``` shell
# CPU-only version
$ conda install -c pytorch faiss-cpu=1.10.0

# GPU(+CPU) version
$ conda install -c pytorch -c nvidia faiss-gpu=1.10.0

# GPU(+CPU) version with NVIDIA cuVS
$ conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.10.0

# GPU(+CPU) version using AMD ROCm not yet available
```

For faiss-gpu, the nvidia channel is required for CUDA, which is not published in the main anaconda channel.

For faiss-gpu-cuvs, the rapidsai, conda-forge and nvidia channels are required.

Nightly pre-release packages can be installed as follows:

``` shell
# CPU-only version
$ conda install -c pytorch/label/nightly faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.10.0

# GPU(+CPU) version with NVIDIA cuVS (package built with CUDA 12.4)
conda install -c pytorch -c rapidsai -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5'

# GPU(+CPU) version with NVIDIA cuVS (package built with CUDA 11.8)
conda install -c pytorch -c rapidsai -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=11.4,<=11.8'

# GPU(+CPU) version using AMD ROCm not yet available
```
In the above commands, pytorch-cuda=11 or pytorch-cuda=12 would select a specific CUDA version, if it’s required.

A combination of versions that installs GPU Faiss with CUDA and Pytorch (as of 2024-05-15):
```
conda create --name faiss_1.8.0
conda activate faiss_1.8.0
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=11 numpy
```

## Installing from conda-forge

Faiss is also being packaged by [conda-forge](https://conda-forge.org/), the
community-driven packaging ecosystem for conda. The packaging effort is
collaborating with the Faiss team to ensure high-quality package builds.

Due to the comprehensive infrastructure of conda-forge, it may even happen that
certain build combinations are supported in conda-forge that are not available
through the pytorch channel. To install, use

``` shell
# CPU version
$ conda install -c conda-forge faiss-cpu

# GPU version
$ conda install -c conda-forge faiss-gpu

# NVIDIA cuVS and AMD ROCm version not yet available
```

You can tell which channel your conda packages come from by using `conda list`.
If you are having problems using a package built by conda-forge, please raise
an [issue](https://github.com/conda-forge/faiss-split-feedstock/issues) on the
conda-forge package "feedstock".

# Building from source

Faiss can be built from source using CMake.

Faiss is supported on x86-64 machines on Linux, OSX, and Windows. It has been
found to run on other platforms as well, see
[other platforms](https://github.com/facebookresearch/faiss/wiki/Related-projects#bindings-to-other-languages-and-porting-to-other-platforms).

The basic requirements are:
- a C++17 compiler (with support for OpenMP support version 2 or higher),
- a BLAS implementation (on Intel machines we strongly recommend using Intel MKL for best
performance).

The optional requirements are:
- for GPU indices:
  - nvcc,
  - the CUDA toolkit,
- for AMD GPUs:
  - AMD ROCm,
- for using NVIDIA cuVS implementations:
  - libcuvs=24.12
- for the python bindings:
  - python 3,
  - numpy,
  - and swig.

Indications for specific configurations are available in the [troubleshooting
section of the wiki](https://github.com/facebookresearch/faiss/wiki/Troubleshooting).

### Building with NVIDIA cuVS

[cuVS](https://docs.rapids.ai/api/cuvs/nightly/) contains state-of-the-art implementations of several algorithms for running approximate nearest neighbors and clustering on the GPU. It is built on top of the [RAPIDS RAFT](https://github.com/rapidsai/raft) library of high performance machine learning primitives. Building Faiss with cuVS enabled allows a user to choose between regular GPU implementations in Faiss and cuVS implementations for specific algorithms.

The libcuvs dependency should be installed via conda:
1. With CUDA 12.0 - 12.5:
```
conda install -c rapidsai -c conda-forge -c nvidia libcuvs=24.12 'cuda-version>=12.0,<=12.5'
```
2. With CUDA 11.4 - 11.8
```
conda install -c rapidsai -c conda-forge -c nvidia libcuvs=24.12 'cuda-version>=11.4,<=11.8'
```
For more ways to install cuVS 24.12, refer to the [RAPIDS Installation Guide](https://docs.rapids.ai/install).

## Step 1: invoking CMake

``` shell
$ cmake -B build .
```

This generates the system-dependent configuration/build files in the `build/`
subdirectory.

Several options can be passed to CMake, among which:
- general options:
  - `-DFAISS_ENABLE_GPU=OFF` in order to disable building GPU indices (possible
  values are `ON` and `OFF`),
  - `-DFAISS_ENABLE_PYTHON=OFF` in order to disable building python bindings
  (possible values are `ON` and `OFF`),
  - `-DFAISS_ENABLE_CUVS=ON` in order to use the NVIDIA cuVS implementations
    of the IVF-Flat, IVF-PQ and [CAGRA](https://arxiv.org/pdf/2308.15136) GPU-accelerated indices (default is `OFF`, possible, values are `ON` and `OFF`).
    Note: `-DFAISS_ENABLE_GPU` must be set to `ON` when enabling this option.
  - `-DBUILD_TESTING=OFF` in order to disable building C++ tests,
  - `-DBUILD_SHARED_LIBS=ON` in order to build a shared library (possible values
  are `ON` and `OFF`),
  - `-DFAISS_ENABLE_C_API=ON` in order to enable building [C API](c_api/INSTALL.md) (possible values
    are `ON` and `OFF`),
- optimization-related options:
  - `-DCMAKE_BUILD_TYPE=Release` in order to enable generic compiler
  optimization options (enables `-O3` on gcc for instance),
  - `-DFAISS_OPT_LEVEL=avx2` in order to enable the required compiler flags to
  generate code using optimized SIMD/Vector instructions. Possible values are below:
    - On x86-64, `generic`, `avx2`, 'avx512', and `avx512_spr` (for avx512 features available since Intel(R) Sapphire Rapids), by increasing order of optimization,
    - On aarch64, `generic` and `sve`, by increasing order of optimization,
  - `-DFAISS_USE_LTO=ON` in order to enable [Link-Time Optimization](https://en.wikipedia.org/wiki/Link-time_optimization) (default is `OFF`, possible values are `ON` and `OFF`).
- BLAS-related options:
  - `-DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=/path/to/mkl/libs` to use the
  Intel MKL BLAS implementation, which is significantly faster than OpenBLAS
  (more information about the values for the `BLA_VENDOR` option can be found in
  the [CMake docs](https://cmake.org/cmake/help/latest/module/FindBLAS.html)),
- GPU-related options:
  - `-DCUDAToolkit_ROOT=/path/to/cuda-10.1` in order to hint to the path of
  the CUDA toolkit (for more information, see
  [CMake docs](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)),
  - `-DCMAKE_CUDA_ARCHITECTURES="75;72"` for specifying which GPU architectures
  to build against (see [CUDA docs](https://developer.nvidia.com/cuda-gpus) to
  determine which architecture(s) you should pick),
  - `-DFAISS_ENABLE_ROCM=ON` in order to enable building GPU indices for AMD GPUs.
 `-DFAISS_ENABLE_GPU` must be `ON` when using this option. (possible values are `ON` and `OFF`),
- python-related options:
  - `-DPython_EXECUTABLE=/path/to/python3.7` in order to build a python
  interface for a different python than the default one (see
  [CMake docs](https://cmake.org/cmake/help/latest/module/FindPython.html)).

## Step 2: Invoking Make

``` shell
$ make -C build -j faiss
```

This builds the C++ library (`libfaiss.a` by default, and `libfaiss.so` if
`-DBUILD_SHARED_LIBS=ON` was passed to CMake).

The `-j` option enables parallel compilation of multiple units, leading to a
faster build, but increasing the chances of running out of memory, in which case
it is recommended to set the `-j` option to a fixed value (such as `-j4`).

If making use of optimization options, build the correct target before swigfaiss.

For AVX2:

``` shell
$ make -C build -j faiss_avx2
```

For AVX512:

``` shell
$ make -C build -j faiss_avx512
```

For AVX512 features available since Intel(R) Sapphire Rapids.

``` shell
$ make -C build -j faiss_avx512_spr
```

This will ensure the creation of neccesary files when building and installing the python package.

## Step 3: Building the python bindings (optional)

``` shell
$ make -C build -j swigfaiss
$ (cd build/faiss/python && python setup.py install)
```

The first command builds the python bindings for Faiss, while the second one
generates and installs the python package.


## Step 4: Installing the C++ library and headers (optional)

``` shell
$ make -C build install
```

This will make the compiled library (either `libfaiss.a` or `libfaiss.so` on
Linux) available system-wide, as well as the C++ headers. This step is not
needed to install the python package only.


## Step 5: Testing (optional)

### Running the C++ test suite

To run the whole test suite, make sure that `cmake` was invoked with
`-DBUILD_TESTING=ON`, and run:

``` shell
$ make -C build test
```

### Running the python test suite

``` shell
$ (cd build/faiss/python && python setup.py build)
$ PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
```

### Basic example

A basic usage example is available in
[`demos/demo_ivfpq_indexing.cpp`](https://github.com/facebookresearch/faiss/blob/main/demos/demo_ivfpq_indexing.cpp).

It creates a small index, stores it and performs some searches. A normal runtime
is around 20s. With a fast machine and Intel MKL's BLAS it runs in 2.5s.

It can be built with
``` shell
$ make -C build demo_ivfpq_indexing
```
and subsequently ran with
``` shell
$ ./build/demos/demo_ivfpq_indexing
```

### Basic GPU example

``` shell
$ make -C build demo_ivfpq_indexing_gpu
$ ./build/demos/demo_ivfpq_indexing_gpu
```

This produce the GPU code equivalent to the CPU `demo_ivfpq_indexing`. It also
shows how to translate indexes from/to a GPU.

### A real-life benchmark

A longer example runs and evaluates Faiss on the SIFT1M dataset. To run it,
please download the ANN_SIFT1M dataset from http://corpus-texmex.irisa.fr/
and unzip it to the subdirectory `sift1M` at the root of the source
directory for this repository.

Then compile and run the following (after ensuring you have installed faiss):

``` shell
$ make -C build demo_sift1M
$ ./build/demos/demo_sift1M
```

This is a demonstration of the high-level auto-tuning API. You can try
setting a different index_key to find the indexing structure that
gives the best performance.

### Real-life test

The following script extends the demo_sift1M test to several types of
indexes. This must be run from the root of the source directory for this
repository:

``` shell
$ mkdir tmp  # graphs of the output will be written here
$ python demos/demo_auto_tune.py
```

It will cycle through a few types of indexes and find optimal
operating points. You can play around with the types of indexes.

### Real-life test on GPU

The example above also runs on GPU. Edit `demos/demo_auto_tune.py` at line 100
with the values

``` python
keys_to_test = keys_gpu
use_gpu = True
```

and you can run
``` shell
$ python demos/demo_auto_tune.py
```
to test the GPU code.

[^1]: The vector search and clustering algorithms in NVIDIA RAFT have been formally migrated to [NVIDIA cuVS](https://github.com/rapidsai/cuvs). This package is being renamed to `faiss-gpu-cuvs` in the next stable release, which will use these GPU implementations from the pre-compiled `libcuvs=24.12` binary.
