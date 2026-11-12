#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


# Build swigfaiss.so
# Metal bindings are enabled on macOS where faiss_metal was built.
METAL_FLAG=""
if [ "$(uname -s)" = "Darwin" ]; then
  METAL_FLAG="-DFAISS_ENABLE_METAL=ON"
fi

cmake -B _build_python_${PY_VER} \
      -Dfaiss_ROOT=_libfaiss_stage/ \
      -DFAISS_ENABLE_GPU=OFF \
      $METAL_FLAG \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      faiss/python

make -C _build_python_${PY_VER} -j$(nproc) swigfaiss

# Build actual python module.
cd _build_python_${PY_VER}/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX

# Copy MetalDistance.metallib into the installed Python package directory so
# the dladdr-based runtime lookup in MetalKernels.mm finds it next to
# swigfaiss.so. MetalKernels.mm uses dladdr to get the directory of the .so
# (which is $SP_DIR/faiss/) and then appends "MetalDistance.metallib" -- so
# the metallib must live alongside the Python extension, not just in
# $PREFIX/lib/ where cmake installs it.
if [ "$(uname -s)" = "Darwin" ] && [ -f ../_libfaiss_stage/lib/MetalDistance.metallib ]; then
  cp ../_libfaiss_stage/lib/MetalDistance.metallib "$SP_DIR/faiss/"
fi
