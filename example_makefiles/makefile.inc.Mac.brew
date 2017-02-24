
# -*- makefile -*-

# Tested on macOS Sierra (10.12.2) with llvm installed using Homebrew (https://brew.sh)
# brew install llvm
CC=/usr/local/opt/llvm/bin/clang++
CFLAGS=-fPIC -m64 -Wall -g -O3 -msse4 -mpopcnt -fopenmp -Wno-sign-compare -Dnullptr=NULL -I/usr/local/opt/llvm/include -Doverride=
LDFLAGS=-g -fPIC -fopenmp -L/usr/local/opt/llvm/lib

# common mac flags
SHAREDEXT=dylib
SHAREDFLAGS=-Wl,-F. -bundle -undefined dynamic_lookup
FAISSSHAREDFLAGS=-dynamiclib

#  wrapldflags=""
#  sharedext=dylib
#  sharedflags="-dynamiclib"
#  yaelsharedflags="$sharedflags -install_name $yaelprefix/yael/libyael.dylib"

##########################################################################
# Uncomment one of the 4 BLAS/Lapack implementation options
# below. They are sorted # from fastest to slowest (in our
# experiments).
##########################################################################

#
# 1. Intel MKL
#
# This is the fastest BLAS implementation we tested. Unfortunately it
# is not open-source and determining the correct linking flags is a
# nightmare. See
#
#   https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
#
# for a start on setting the link flags. On version IntelComposerXE
# 2015.0.090, the following flags work
#
# MKLROOT=$(HOME)/fbsource/fbcode/third-party2//IntelComposerXE/2015.0.090/gcc-4.8.1-glibc-2.17/c3f970a/mkl
#
# BLASLDFLAGS=-Wl,--no-as-needed -L$(MKLROOT)/lib/intel64   -lmkl_intel_ilp64 \
# -lmkl_core -lmkl_gnu_thread -ldl -lpthread
#
# the ilp64 means that the integers are 64-bit.
#
# BLASLDFLAGS=-DFINTEGER=long
#
# you may have to set the LD_LIBRARY_PATH=$MKLROOT/lib/intel64 at runtime
#

#
# 2. Openblas
#
# The library contains both BLAS and Lapack. Install with port install OpenBLAS
#
# BLASCFLAGS=-DFINTEGER=int
# BLASLDFLAGS=/opt/local/lib/libopenblas.dylib
#

#
# 3. Apple's framework accelerate
#
# This has the advantage that it does not require to install anything,
# as it is provided by default on the mac. It is not very fast, though.
#

BLASCFLAGS=-DFINTEGER=int
BLASLDFLAGS=-framework Accelerate



##########################################################################
# SWIG and Python flags
##########################################################################

# SWIG executable. This should be at least version 3.x
# brew install swig

SWIGEXEC=/usr/local/bin/swig

# The Python include directories for the current python executable can
# typically be found with
#
# python -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()"
# python -c "import numpy ; print numpy.get_include()"
#
# the paths below are for the system python (not the macports one)

PYTHONCFLAGS=-I/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
-I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include


##########################################################################
# Faiss GPU
##########################################################################

# As we don't have access to a Mac with nvidia GPUs installed, we
# could not validate the GPU compile of Faiss.
