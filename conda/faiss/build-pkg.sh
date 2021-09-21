#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


# Build swigfaiss.so/swigfaiss_avx2.so.
cmake -B _build_python_${PY_VER} \
      -Dfaiss_ROOT=_libfaiss_stage/ \
      -DFAISS_OPT_LEVEL=avx2 \
      -DFAISS_ENABLE_GPU=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      faiss/python

make -C _build_python_${PY_VER} -j $CPU_COUNT swigfaiss swigfaiss_avx2

# Build actual python module.
cd _build_python_${PY_VER}/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX
