# Installing Faiss via conda

The supported way to install Faiss is through [conda](https://docs.conda.io).
Stable releases are pushed regularly to the pytorch conda channel, as well as
pre-release nightly builds.

- The CPU-only faiss-cpu conda package is currently available on Linux (x86-64 and aarch64), OSX (arm64 only), and Windows (x86-64)
- faiss-gpu, containing both CPU and GPU indices, is available on Linux (x86-64 only) for CUDA 11.4 and 12.1
- faiss-gpu-cuvs package containing GPU indices provided by [NVIDIA cuVS](https://github.com/rapidsai/cuvs/) version 26.06, is available on Linux (x86-64 only) for CUDA 13.2.

To install the latest stable release:

``` shell
# CPU-only version
$ conda install -c pytorch -c conda-forge faiss-cpu=1.15.1

# GPU(+CPU) version
$ conda install -c pytorch -c nvidia -c conda-forge faiss-gpu=1.15.1

# GPU(+CPU) version with NVIDIA cuVS
$ conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.15.1

# GPU(+CPU) version using AMD ROCm not yet available
```

The conda-forge channel is required for up-to-date dependencies (MKL on x86-64, OpenBLAS on ARM), which are not regularly updated in the default Anaconda channel.

For faiss-gpu, the nvidia channel is additionally required for CUDA, which is not published in the main anaconda channel.

For faiss-gpu-cuvs, the rapidsai, conda-forge and nvidia channels are required.

If you prefer [Pixi](https://pixi.sh/latest/), you can use the same channels and package names with a Conda-compatible workflow:

``` shell
# CPU-only version
$ pixi init -c pytorch -c conda-forge
$ pixi add faiss-cpu=1.15.1

# GPU(+CPU) version
$ pixi init -c pytorch -c nvidia -c conda-forge
$ pixi add faiss-gpu=1.15.1

# GPU(+CPU) version with NVIDIA cuVS
$ pixi init -c pytorch -c nvidia -c rapidsai -c conda-forge
$ pixi add libnvjitlink faiss-gpu-cuvs=1.15.1
```

Pixi resolves packages from the listed Conda channels, so the same channel requirements described above still apply.

Nightly pre-release packages can be installed as follows:

``` shell
# CPU-only version
$ conda install -c pytorch/label/nightly -c conda-forge faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch/label/nightly -c nvidia -c conda-forge faiss-gpu=1.15.1

# GPU(+CPU) version with NVIDIA cuVS (package built with CUDA 13.2)
conda install -c pytorch -c rapidsai -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version=13.2'

# GPU(+CPU) version using AMD ROCm not yet available
```
In the above commands, pytorch-cuda=11 or pytorch-cuda=12 would select a specific CUDA version, if it’s required.

A combination of versions that installs GPU Faiss with CUDA and Pytorch (as of 2024-05-15):
```
conda create --name faiss_1.8.0
conda activate faiss_1.8.0
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=11 numpy
```

# Building from source

Faiss can be built from source using CMake.

Faiss is supported on x86-64 machines on Linux, OSX, and Windows. It has been
found to run on other platforms as well, see
[other platforms](https://github.com/facebookresearch/faiss/wiki/Related-projects#bindings-to-other-languages-and-porting-to-other-platforms).

The basic requirements are:
- a C++20 compiler (with OpenMP support version 2 or higher),
- a BLAS implementation (on Intel machines we strongly recommend using Intel MKL for best
performance).

The optional requirements are:
- for GPU indices:
  - nvcc,
  - the CUDA toolkit,
- for AMD GPUs:
  - AMD ROCm,
- for using NVIDIA cuVS implementations:
  - libcuvs=26.10
- for the python bindings:
  - python 3,
  - numpy,
  - and swig.

Indications for specific configurations are available in the [troubleshooting
section of the wiki](https://github.com/facebookresearch/faiss/wiki/Troubleshooting).

### Building with NVIDIA cuVS

[cuVS](https://docs.rapids.ai/api/cuvs/stable/) contains state-of-the-art implementations of several algorithms for running approximate nearest neighbors and clustering on the GPU. It is built on top of the [RAPIDS RAFT](https://github.com/rapidsai/raft) library of high performance machine learning primitives. Building Faiss with cuVS enabled allows a user to choose between regular GPU implementations in Faiss and cuVS implementations for specific algorithms.

The libcuvs dependency should be installed via conda:
```
conda install -c rapidsai -c conda-forge -c nvidia libcuvs=26.10 'cuda-version=13.2'
```
For more ways to install cuVS 26.10, refer to the [RAPIDS Installation Guide](https://docs.rapids.ai/install).

### Building with Intel(R) SVS

[Intel(R) Scalable Vector Search (SVS)](https://github.com/intel/ScalableVectorSearch) is a library for high-performance vector search. Building Faiss with SVS enabled allows using SVS implementations of graph-based indices (e.g., Vamana).

The SVS library will be automatically fetched and built by CMake if `FAISS_ENABLE_SVS` is set to `ON`.

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
  - `-DFAISS_ENABLE_SVS=ON` in order to enable the Intel(R) Scalable Vector Search (SVS) integration (default is `OFF`, possible values are `ON` and `OFF`).
    Note: This will download and build the SVS runtime library (`libsvs_runtime.so`). When installing the python package, this library will be copied into the package directory. For C++ usage, ensure this library is in your library path.
- optimization-related options:
  - `-DCMAKE_BUILD_TYPE=Release` in order to enable generic compiler
  optimization options (enables `-O3` on gcc for instance),
  - `-DFAISS_OPT_LEVEL=avx2` in order to enable the required compiler flags to
  generate code using optimized SIMD/Vector instructions. Possible values are below:
    - On x86-64, `generic`, `avx2`, `avx512`, and `avx512_spr` (for avx512 features available since Intel(R) Sapphire Rapids), by increasing order of optimization,
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

This will ensure the creation of necessary files when building and installing the python package.

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

This produces the GPU code equivalent to the CPU `demo_ivfpq_indexing`. It also
shows how to translate indexes from/to a GPU.

### CAGRA datasets and HNSW interop

With `FAISS_ENABLE_CUVS=ON`, CAGRA builds require a caller-owned
`faiss::gpu::Dataset`. The index owns its graph and borrows vector storage.
`viewHost`, `viewDevice`, and `viewDevicePadded` borrow an existing allocation;
`makeDevicePadded` delegates allocation and padding to cuVS. Like cuVS, it rejects
device/managed input whose row stride already matches the required padding;
use a view for that storage. Preparation runs asynchronously on the resource's
stream, so keep the source allocation alive and unchanged until copying finishes.
The returned Dataset controls the new allocation's lifetime. Calling
`dataset.view()` creates another non-owning descriptor of the same storage
without extending its lifetime.
Public signatures take pointers, extents, strides, and Faiss types, without
requiring RAFT or cuVS types.

For a packed host Float32 array `xb` with `n` rows and dimension `d`, build a graph
and export independent CPU HNSW storage as follows:

```cpp
#include <faiss/IndexHNSW.h>
#include <faiss/gpu/Dataset.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/StandardGpuResources.h>

faiss::gpu::StandardGpuResources resources;
auto source = faiss::gpu::Dataset::viewHost(
        xb, n, d, faiss::NumericType::Float32, d);
faiss::gpu::GpuIndexCagra graphOnly(&resources, d);
graphOnly.train(source); // Host data builds a graph; isSearchReady() is false.

faiss::IndexHNSWCagra cpu(d, 32);
cpu.base_level_only = true;
graphOnly.copyTo(&cpu); // CPU index now owns its graph and packed vectors.

faiss::gpu::GpuIndexCagra imported(&resources, d);
imported.copyFrom(&cpu); // Imports graph and metadata; isSearchReady() is false.
```

The CPU index remains independent of the original Dataset after `copyTo`.
`copyFrom(cpuIndex, Dataset* dataset = nullptr)` imports only the graph and
metadata when the optional Dataset is omitted. It retains no CPU vector storage,
and the CPU index can be destroyed after the call. Graph retrieval remains
available; search and full CPU HNSW export with `copyTo` require attached vectors.

To import a searchable index, wrap a preallocated writable device buffer with
the regular Dataset API and pass its address. For this Float32 example,
`deviceBuffer` must point to at least `n * stride` floats on device 0, with a
16-byte-aligned base address:

```cpp
const faiss::idx_t stride = ((static_cast<faiss::idx_t>(d) + 3) / 4) * 4;
auto destination = faiss::gpu::Dataset::viewDevicePadded(
        deviceBuffer, n, d, faiss::NumericType::Float32, stride, 0);
faiss::gpu::GpuIndexCagra searchable(&resources, d);
searchable.copyFrom(&cpu, &destination); // Fills the buffer; isSearchReady() is true.
```

The supplied Dataset must have the CPU index's row count, dimension, and dtype,
and be CAGRA-compatible on the GPU index's device. A compatible standard-layout
device view is also accepted. `copyFrom` copies the CPU vectors into that
allocation and zeroes its padding; previous contents need not be initialized.
The upload completes before return, so the CPU index can then be destroyed.
Ownership stays with the caller, and the GPU index borrows the allocation for
search. `isSearchReady()` derives readiness from the graph and attached Dataset;
there is no separate readiness setting. Keep the allocation alive and unchanged
through index use and queued GPU work.
`GpuIndexBinaryCagra::copyFrom` follows the same optional-output contract with
`UInt8` and dimensions measured in bytes.

A standard device view is searchable without a vector copy when its base address
is 16-byte aligned and its row stride equals the dimension rounded up to a
16-byte row width. For example, packed Float32 vectors of dimension 12 need no
extra padding. Any padding elements must already be zero. Otherwise, build the
graph from a standard Dataset and explicitly attach prepared storage:

```cpp
auto padded = faiss::gpu::Dataset::makeDevicePadded(&resources, source, 0);
graphOnly.updateDataset(padded);
```

`updateDataset` preserves the graph and borrows already-populated storage;
it does not fill the allocation. The new storage must contain the same vectors
in the same order, shape, and dtype. It checks metadata but cannot verify
vector identity. It can also attach vectors after a graph-only `copyFrom`.
Keep all borrowed allocations alive and unchanged through
index use and queued GPU work; make data ready on the index resource's stream
before use. Caller-supplied CUDA streams must also outlive GPU resources that
cache cuVS allocations, even after reverting the default stream. `reset()` clears
the borrowed binding and allows rebuilding. Repeated `train` or `add` calls on
an already trained CAGRA leave its graph and Dataset unchanged. A moved owning Dataset retains the
allocation and existing bindings. Dimensions and row
strides are in elements; binary CAGRA uses `UInt8` with Dataset dimension equal
to the index's bit dimension divided by eight.

Building a graph with `train` or `add` requires an explicit Dataset. Their
inherited pointer overloads reject calls; use the Dataset
overloads on `GpuIndexCagra` or `GpuIndexBinaryCagra`. Dataset ownership replaces
`store_dataset`. Pointer APIs for other index families are unchanged. Dataset
construction, build, attachment, and import are available in C++ and Python;
there is no C Dataset binding. Existing CPU HNSW search and serialization remain
available. Generic CPU-to-GPU CAGRA cloning imports the graph only, like
`copyFrom(cpu)`; call `updateDataset` with populated device storage before search.

In Python, `Dataset.viewHost(array)` borrows a two-dimensional NumPy array and
infers its shape, dtype, and row stride. It accepts `float32`, `float16`, `int8`,
and `uint8`, with contiguous columns, without copying or converting the array.
For example, binary CAGRA uses a `uint8` array with a dimension in bytes:

```python
import faiss
import numpy as np

xb = np.random.RandomState(7).randint(0, 256, (512, 19)).astype("uint8")
resources = faiss.StandardGpuResources()
source = faiss.Dataset.viewHost(xb)
storage = faiss.Dataset.makeDevicePadded(resources, source, 0)
resources.syncDefaultStreamCurrentDevice()  # xb must survive the async copy.

config = faiss.GpuIndexCagraConfig()
config.build_algo = faiss.graph_build_algo_NN_DESCENT
config.graph_degree = 16
config.intermediate_graph_degree = 32
index = faiss.GpuIndexBinaryCagra(resources, xb.shape[1] * 8, config)
index.train(storage)  # add(storage) also builds the entire graph.
assert index.isSearchReady()
distances, labels = index.search(xb[:8], 1)
packed_copy = storage.copyToHost()  # An independent, ready NumPy array.

cpu = faiss.IndexBinaryHNSWCagra()
cpu.base_level_only = True
index.copyTo(cpu)
imported = faiss.GpuIndexBinaryCagra(resources, index.d, config)
imported.copyFrom(cpu)  # Graph and metadata only; not search ready.
imported.updateDataset(storage)  # Borrows these already-populated vectors.
```

For custom IDs, wrap an empty CAGRA in `IndexIDMap` (or `IndexBinaryIDMap`),
then call `mapped.train(storage)` and `mapped.add_with_ids(storage, ids)`.
The IDs must cover every vector in the graph. `IndexIDMap2` variants are also
supported. The wrapper keeps Dataset ownership on the wrapped CAGRA, so deleting
the IDMap does not invalidate a CAGRA that the caller still retains.

Python retains references to keep borrowed storage alive: `viewHost(array)`
retains the NumPy array, `view()` retains its source Dataset, and CAGRA retains
the Dataset passed to `train`, `add`, `updateDataset`, or `copyFrom`.
Deleting the user's Dataset variable therefore does not invalidate a search-ready
index. Successful replacement, graph-only import, or `reset()` releases the index's
previous Dataset reference; destroying the index also releases it. An owning
Dataset frees its allocation when its last reference is released, including
references from other indexes or views. C++ ownership and stream rules remain
unchanged: preparation is asynchronous, so retain the source until its copy
finishes. Raw-pointer views cannot retain an allocation owner; retain that owner
explicitly. `copyToHost()` waits for device work and returns an independent,
packed NumPy array with the original dtype. For Float16 or Int8 indexes, pass
the matching search type:
`index.search(xq, k, numeric_type=storage.getNumericType())`.

Use `copyFrom(cpu)` or `copyFrom(cpu, None)` for graph-only import, and
`copyFrom(cpu, destination)` to fill and attach existing device storage under
the contract above. The raw `Dataset.viewDevice` and `Dataset.viewDevicePadded`
argument order is `(data, numRows, dim, numericType, rowStride, device)`.
Convert an integer device address with `faiss.cast_integer_to_void_ptr(address)`
when passing `data`. Raw host-pointer and output operations remain available as
`viewHost_c` and `copyToHost_c`.

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
