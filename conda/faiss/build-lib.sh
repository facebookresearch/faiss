# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

if [ $1 = "cpu" ]; then GPU=OFF; else GPU=ON; fi


# Build libfaiss_avx2.so.
cmake -B _build_avx2 \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=avx2 \
      -DFAISS_ENABLE_GPU=$GPU \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_BUILD_TYPE=Release .

cmake --build _build_avx2 -j $CPU_COUNT

cmake --install _build_avx2 --prefix $PREFIX
cmake --install _build_avx2 --prefix _libfaiss_avx2_stage/


# Build libfaiss.so.
cmake -B _build_generic \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_ENABLE_GPU=$GPU \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_BUILD_TYPE=Release .

cmake --build _build_generic -j $CPU_COUNT

cmake --install _build_generic --prefix $PREFIX
cmake --install _build_generic --prefix _libfaiss_stage/
