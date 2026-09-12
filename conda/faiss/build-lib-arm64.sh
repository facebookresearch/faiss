#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


# Build libfaiss.so
# Use Apple Accelerate for BLAS/LAPACK on arm64 macOS instead of OpenBLAS.
# Without an explicit vendor, cmake may fail to locate BLAS when openblas
# is absent from the conda build environment.
cmake -B _build \
      -DFAISS_ENABLE_MKL=OFF \
      -DBLA_VENDOR=Apple \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_ENABLE_C_API=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=sve \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release .

if [ "$(uname -s)" = "Darwin" ]; then
  make -C _build -j"$(nproc)" faiss faiss_sve faiss_c faiss_c_sve faiss_metal metal_shaders
else
  make -C _build -j"$(nproc)" faiss faiss_sve faiss_c faiss_c_sve
fi

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _libfaiss_stage/
