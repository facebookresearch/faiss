
[//]: # "**********************************************************"
[//]: # "** INSTALL file for Faiss (Fair AI Similarity Search    **"
[//]: # "**********************************************************"

INSTALL file for Faiss (Fair AI Similarity Search)
==================================================

The Faiss installation works in 3 steps, from easiest to most
involved:

1. compile the C++ core and examples

2. compile the Python interface

3. compile GPU part

Steps 2 and 3 depend on 1, but they are otherwise independent.

Alternatively, all 3 steps above can be run by building a Docker image (see section "Docker instructions" below).

General compilation instructions
================================

Faiss has been tested only on x86_64 machines on Linux and Mac OS.

Faiss is compiled via a Makefile. The system-dependent configuration
of the Makefile is in an include file, makefile.inc. The variables in
makefile.inc must be set by hand.

Faiss requires a C++ compiler that understands:
- the Intel intrinsics for SSE instructions
- the GCC intrinsic for the popcount instruction
- basic OpenMP

There are a few models for makefile.inc in the example_makefiles/
subdirectory. Copy the relevant one for your system and adjust to your
needs. There are also indications for specific configurations in the
troubleshooting section of the wiki.

https://github.com/facebookresearch/faiss/wiki/Troubleshooting

Faiss comes as a .a archive, that can be linked with executables or
dynamic libraries (useful for the Python wrapper).


Step 1: Compiling the C++ Faiss
===============================

The CPU version of Faiss is written in C++03, so it should compile
even with relatively old C++ compilers.

BLAS/Lapack
-----------

The only variables that need to be configured for the C++ Faiss are
the BLAS/Lapack flags (a linear aglebra software package). It needs a
flag telling whether BLAS/Lapack uses 32 or 64 bit integers and the
linking flags. Faiss uses the Fortran 77 interface of BLAS/Lapack and
thus does not need an include path.

There are several BLAS implementations, depending on the OS and
machine. To have reasonable performance, the BLAS library should be
multithreaded. See the example makefile.inc's for hints and examples
on how to set the flags.

To check that the link flags are correct, and verify whether the
implementation uses 32 or 64 bit integers, you can

  `make tests/test_blas`

and run

  `./tests/test_blas`

Testing Faiss
-------------

Once the proper BLAS flags are set, the library should compile
smoothly by running

  `make`

A basic usage example is in

  `tests/demo_ivfpq_indexing`

it makes a small index, stores it and performs some searches. A normal
runtime is around 20s. With a fast machine and Intel MKL's BLAS it
runs in 2.5s.

A real-life benchmark
---------------------

A bit longer example runs and evaluates Faiss on the SIFT1M
dataset. To run it, please download the ANN_SIFT1M dataset from

http://corpus-texmex.irisa.fr/

and unzip it to the sudirectory sift1M.

Then compile and run

```
make tests/demo_sift1M
tests/demo_sift1M
```

This is a demonstration of the high-level auto-tuning API. You can try
setting a different index_key to find the indexing structure that
gives the best performance.


Step 2: Compiling the Python interface
======================================

The Python interface is compiled with

  `make py`

If you want to compile it for another python version than the default
Python 2.7, in particular Python 3, the PYTHONCFLAGS must be adjusted in
makefile.inc, see the examples.

How it works
------------

The Python interface is provided via SWIG (Simple Wrapper and
Interface Generator) and an additional level of manual wrappers (in faiss.py).

SWIG generates two wrapper files: a Python file (`swigfaiss.py`) and a
C++ file that must be compiled to a dynamic library (`_swigfaiss.so`). These
files are included in the repository, so running swig is only required when
the C++ headers of Faiss are changed.

The C++ compilation to the dynamic library requires to set:

- `SHAREDFLAGS`: system-specific flags to generate a dynamic library

- `PYTHONCFLAGS`: include flags for Python

See the example makefile.inc's on how to set the flags.


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
indexes:

```
export PYTHONPATH=.   # needed because the script is in a subdirectory
mkdir tmp             # some output will be written there
python python/demo_auto_tune.py
```

It will cycle through a few types of indexes and find optimal
operating points. You can play around with the types of indexes.


Step 3: Compiling the GPU implementation
========================================

There is a GPU-specific Makefile in the `gpu/` directory. It depends on
the same ../makefile.inc for system-specific variables. You need
libfaiss.a from Step 1 for this to work.

The GPU version is a superset of the CPU version. In addition it
requires:

- a C++11 compliant compiler (and flags)

- the cuda compiler and related libraries (Cublas)

See the example makefile on how to set the flags.

The nvcc-specific flags to pass to the compiler, based on your desired
compute capability. Only compute capability 3.5+ is supported. For
example, we enable by default:

```
-gencode arch=compute_35,code="compute_35"
-gencode arch=compute_52,code="compute_52"
-gencode arch=compute_60,code="compute_60"
```

However, look at https://developer.nvidia.com/cuda-gpus to determine
what compute capability you need to use, and replace our gencode
specifications with the one(s) you need.

Most other flags are related to the C++11 compiler used by nvcc to
complile the actual C++ code. They are normally just transmitted by
nvcc, except some of them that are not recognized and that should be
escaped by prefixing them with -Xcompiler. Also link flags that are
prefixed with -Wl, should be passed with -Xlinker.

Then compile with

  `cd gpu; make`

You may want to add `-j 10` to use 10 threads during compile.

Testing the GPU implementation
------------------------------

Compile the example with

  `cd gpu; make test/demo_ivfpq_indexing_gpu`

This produce the GPU code equivalent to the CPU
demo_ivfpq_indexing. It also shows how to translate indexed from/to
the GPU.

Compiling the Python interface with GPU support
-----------------------------------------------

Given step 2, adding support of the GPU from Python is quite
straightforward. Run

`cd gpu; make py`

The import is the same for the GPU version and the CPU-only
version.

`python -c "import faiss"`

Faiss tries to load the GPU version first, and in case of failure,
loads the CPU-only version. To investigate more closely the cause of
a failure, you can run:

`python -c "import _swigfaiss_gpu"`

Python example with GPU support
-------------------------------

The auto-tuning example above also runs on the GPU. Edit
`python/demo_auto_tune.py` around line 100 with the values

```python
keys_to_test = keys_gpu
use_gpu = True
```

and you can run

```
export PYTHONPATH=.
python/demo_auto_tune.py
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


Hot to use Faiss in your own projects
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

For binary-only distributions, the include files should be under
a faiss/ directory, so that they can be included as

```c++
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
```

Python
------

To import Faiss in your own Python project, you need the files

```
faiss.py
swigfaiss.py  / swigfaiss_gpu.py
_swigfaiss.so / _swigfaiss_gpu.so
```

to be visible in the PYTHONPATH or in the current directory.
Then Faiss can be used in python with

```python
import faiss
```


CMake build instructions:
=========================
Alternatively, Faiss can be built via the experimental cmake scripts.
The installation process is similar to using Makefiles. After installing
the necessary dependencies (OpenBLAS, OpenMP, and CUDA, if BUILD_WITH_GPU
is enabled), the build process can be done by the following commands:

```
mkdir build
cmake ..
make      # use -j to enable parallel build
```

Notes for build on Mac: The native compiler on Mac does not support OpenMP.
So to make it work on Mac, you have to install a new compiler using either
Macports or Homebrew. For example, after installing the compiler `g++-mp-6`
from Macports (`port install g++-mp-6`), you need to set the two flags
`CMAKE_CXX_COMPILER` and `CMAKE_C_COMPILER`:

`cmake -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-6 -DCMAKE_C_COMPILER=/opt/local/bin/gcc-mp-6  ..`

Similarly, you can use Homebrew to install clang++ (`brew install llvm`) and
then set the two flags to `/usr/local/opt/llvm/bin/clang++`.

CMake supports the OpenBLAS and MKL implementations. CMake limitations: the python interface is
NOT supported at this point.

Use Faiss as a 3rd-party library: Using Faiss as a 3rd-party lib via CMake is easy.
If the parental project is also build via CMake, just add a line `add_subdirectory(faiss)`
in CMake where faiss is the sub-folder name. To link Faiss to your application, use

```
add_executable(my_app my_app.cpp)
target_link_libraries(my_app gpufaise faiss)
```
