[//]: # "**********************************************************"
[//]: # "** INSTALL file for Faiss (Fair AI Similarity Search    **"
[//]: # "**********************************************************"

INSTALL file for Faiss (Fair AI Similarity Search)
==================================================

Install via Conda
-----------------

The easiest way to install FAISS is from Anaconda. We regularly push stable releases to the pytorch conda channel.

Currently we support faiss-cpu both on Linux and OSX. We also provide faiss-gpu compiled with CUDA8/CUDA9/CUDA10 on Linux systems.

You can easily install it by

```
# CPU version only
conda install faiss-cpu -c pytorch

# GPU version
conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8
conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10
```

Compile from source
-------------------

The Faiss compilation works in 2 steps:

1. compile the C++ core and examples

2. compile the Python interface

Steps 2 depends on 1.

It is also possible to build a pure C interface. This optional process is
described separately (please see the [C interface installation file](c_api/INSTALL.md))

General compilation instructions
================================

TL;DR:
```bash
cmake -B build .
make -C build

# For the Python interface:
cd build/faiss/python && python setup.py install
```

1. `cmake`

This generates the system-dependent configuration/build files.

A few useful options:
- `-DFAISS_ENABLE_GPU=OFF` in order to build the CPU part only.
- `-DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=/path/to/mkl/libs` to use the MKL BLAS implementation (otherwise it uses the system BLAS that is usually slow).
- `-DCUDAToolkit_ROOT=/path/to/cuda-10.1` in order to hint to the path of
the cudatoolkit.
- `-DCMAKE_CUDA_ARCHITECTURES="75;72"` for specifying which GPU architectures to build against.
- `-DPython_EXECUTABLE=/path/to/python3.7` in order to build a python
interface for a different python than the default one.

2. `make`

This builds the C++ library (the whole library if a suitable cuda toolkit was
found, or the CPU part only otherwise).

3. `make install` (optional)

This installs the headers and libraries.

Faiss is supported on x86_64 machines on Linux, Mac OS and Windows.
It has been found to run on other platforms as well, see [other platforms](https://github.com/facebookresearch/faiss/wiki/Related-projects#bindings-to-other-languages-and-porting-to-other-platforms)

Faiss requires a C++ compiler that understands:
- the Intel intrinsics for SSE instructions,
- the GCC intrinsic for the popcount and a few other instruction,
- basic OpenMP (2.x).

There are indications for specific configurations in the
troubleshooting section of the wiki.

https://github.com/facebookresearch/faiss/wiki/Troubleshooting

Faiss comes as a .a archive, that can be linked with executables or
dynamic libraries (useful for the Python wrapper).


BLAS/Lapack
-----------

The only variables that need to be configured for the C++ Faiss are
the BLAS/Lapack flags (a linear aglebra software package). It needs a
flag telling whether BLAS/Lapack uses 32 or 64 bit integers and the
linking flags. Faiss uses the Fortran 77 interface of BLAS/Lapack and
thus does not need an include path.

There are several BLAS implementations, depending on the OS and
machine. To have reasonable performance, the BLAS library should be
multithreaded. On Intel platforms, MKL is the fastest BLAS implementation, 
see flags above on how to enable it.


Testing Faiss
-------------

A basic usage example is in

  `demos/demo_ivfpq_indexing.cpp`

which you can build by calling 

```
make -C build demo_ivfpq_indexing
./build/demo_ivfpq_indexing
```

It makes a small index, stores it and performs some searches. A normal
runtime is around 20s. With a fast machine and Intel MKL's BLAS it
runs in 2.5s.

To run the whole test suite, make sure that `cmake` has `-DBUILD_TESTING=ON`:

```
make -C build test
```

A real-life benchmark
---------------------

A bit longer example runs and evaluates Faiss on the SIFT1M
dataset. To run it, please download the ANN_SIFT1M dataset from

http://corpus-texmex.irisa.fr/

and unzip it to the subdirectory `sift1M` at the root of the source
directory for this repository.

Then compile and run the following (after ensuring you have installed faiss):

```
make -C build demo_sift1M
```

This is a demonstration of the high-level auto-tuning API. You can try
setting a different index_key to find the indexing structure that
gives the best performance.


Python Interface
------------

The Python interface is provided via SWIG (Simple Wrapper and
Interface Generator) and an additional level of manual wrappers (in faiss/python/faiss.py).

SWIG generates two wrapper files: a Python file `swigfaiss.py` and a
C++ file that must be compiled to a dynamic library (`_swigfaiss.so`).
There is an AVX2 variant of the files, suffixed with `_avx2`. 

Testing the Python wrapper
--------------------------

Often, a successful compile does not mean that the library works,
because missing symbols are detected only at runtime. You should be
able to load the Faiss dynamic library:

  `python -c "import faiss"`

In case of failure, it reports the first missing symbol. To see all
missing symbols (on Linux), use

  `ldd -r _swigfaiss.so`

Sometimes, problems (eg with BLAS libraries) appear only when actually
calling a BLAS function. A simple way to check this

```python
python -c "import faiss, numpy
faiss.Kmeans(10, 20).train(numpy.random.rand(1000, 10).astype('float32'))"
```

Real-life test
--------------

The following script extends the demo_sift1M test to several types of
indexes.  This must be run from the root of the source directory for this
repository:

```
mkdir tmp             # graphs of the output will be written here
PYTHONPATH=. python demos/demo_auto_tune.py
```

It will cycle through a few types of indexes and find optimal
operating points. You can play around with the types of indexes.

Developing in Faiss
-------------------

To repeatedly compile and run a python test, the following command line is useful: 

```
cmake -B build -DCMAKE_CXX_COMPILER=clang++-8 -DFAISS_ENABLE_GPU=OFF -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=path_to_mkl
 -DPython_EXECUTABLE=$(which python) -DFAISS_OPT_LEVEL=avx2  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON

make -j -C build VERBOSE=1 swigfaiss &&  (cd build/faiss/python/ ; python setup.py build ) && cp contrib/*.py build/faiss/python/build/lib/faiss/contrib/ && (pp=$PWD/build/faiss/python/build/lib;cd tests;  PYTHONPATH=$pp python -m unittest  -v test_index )
```


Step 3: Compiling the GPU implementation
========================================

The GPU version is a superset of the CPU version. In addition it
requires the cuda compiler and related libraries (Cublas)

The nvcc-specific flags to pass to the compiler, based on your desired
compute capability can be customized by providing the `CMAKE_CUDA_ARCHITECTURES`
flag to CMake. Only compute capability 3.5+ is supported. For example, we enable
by default:

```
-gencode=arch=compute_35,code=compute_35
-gencode=arch=compute_52,code=compute_52
-gencode=arch=compute_60,code=compute_60
-gencode=arch=compute_61,code=compute_61
-gencode=arch=compute_70,code=compute_70
-gencode=arch=compute_75,code=compute_75
```

However, look at https://developer.nvidia.com/cuda-gpus to determine
what compute capability you need to use, and replace our gencode
specifications with the one(s) you need.

Most other flags are related to the C++11 compiler used by nvcc to
complile the actual C++ code. They are normally just transmitted by
nvcc, except some of them that are not recognized and that should be
escaped by prefixing them with -Xcompiler. Also link flags that are
prefixed with -Wl, should be passed with -Xlinker.

You may want to add `-j 10` to the `make` command to use 10 threads during compile.

Testing the GPU implementation
------------------------------

Compile the example with

```
make -C build demo_ivfpq_indexing_gpu
```


This produce the GPU code equivalent to the CPU
demo_ivfpq_indexing. It also shows how to translate indexed from/to
the GPU.


Python example with GPU support
-------------------------------

The auto-tuning example above also runs on the GPU. Edit
`demos/demo_auto_tune.py` at line 100 with the values

```python
keys_to_test = keys_gpu
use_gpu = True
```

and you can run

```
export PYTHONPATH=.
python demos/demo_auto_tune.py
```

to test the GPU code.


Docker instructions
===================

For using GPU capabilities of Faiss, you'll need to run "nvidia-docker"
rather than "docker". Make sure that docker
(https://docs.docker.com/engine/installation/) and nvidia-docker
(https://github.com/NVIDIA/nvidia-docker) are installed on your system

To build the "faiss" image, run

  `nvidia-docker build -t faiss .`

or if you don't want/need to clone the sources, just run

  `nvidia-docker build -t faiss github.com/facebookresearch/faiss`

If you want to run the tests during the docker build, uncomment the
last 3 "RUN" steps in the Dockerfile. But you might want to run the
tests by yourself, so just run

  `nvidia-docker run -ti --name faiss faiss bash`

and run what you want. If you need a dataset (like sift1M), download it
inside the created container, or better, mount a directory from the host

  nvidia-docker run -ti --name faiss -v /my/host/data/folder/ann_dataset/sift/:/opt/faiss/sift1M faiss bash


How to use Faiss in your own projects
=====================================

C++
---

The makefile generates a static and a dynamic library

```
libfaiss.a
libfaiss.so (or libfaiss.dylib)
```

the executable should be linked to one of these. If you use
the static version (.a), add the LDFLAGS used in the Makefile.

For binary-only distributions, the headers should be under
a `faiss/` directory, so that they can be included as

```c++
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
```

Python
------

To import Faiss in your own Python project, you need the files

```
__init__.py
swigfaiss.py
_swigfaiss.so
```
to be present in a `faiss/` directory visible in the PYTHONPATH or in the
current directory.
Then Faiss can be used in python with

```python
import faiss
```
