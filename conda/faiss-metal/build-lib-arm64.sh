#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e


# Build libfaiss.so
cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DFAISS_ENABLE_C_API=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=sve \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) faiss faiss_sve faiss_c faiss_c_sve

cmake --install _build --prefix $PREFIX
cmake --install _build --prefix _libfaiss_stage/
