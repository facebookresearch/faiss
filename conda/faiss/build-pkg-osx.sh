#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


# Build swigfaiss.so/swigfaiss_avx2.so/swigfaiss_avx512
cmake -B _build_python_${PY_VER} \
      -Dfaiss_ROOT=_libfaiss_stage/ \
      -DFAISS_OPT_LEVEL=avx512 \
      -DFAISS_ENABLE_GPU=OFF \
      -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 \
      -DOpenMP_CXX_LIB_NAMES=libiomp5 \
      -DOpenMP_libiomp5_LIBRARY=$PREFIX/lib/libiomp5.dylib \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      faiss/python

make -C _build_python_${PY_VER} -j$(nproc) swigfaiss swigfaiss_avx2 swigfaiss_avx512

# Build actual python module.
cd _build_python_${PY_VER}/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX
