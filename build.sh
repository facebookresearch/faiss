#!/bin/bash

BUILD_TYPE=Release

RAFT_REPO_REL="../raft"
RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"

set -e

if [ "$1" == "clean" ]; then
  rm -rf build
  exit 0
fi

if [ "$1" == "test" ]; then
  make -C build -j test
  exit 0
fi

if [ "$1" == "test-raft" ]; then
  ./build/faiss/gpu/test/TestRaftIndexIVFFlat
  exit 0
fi

cmake \
 -DFAISS_ENABLE_GPU=ON \
 -DFAISS_ENABLE_PYTHON=OFF \
 -DBUILD_TESTING=ON \
 -DBUILD_SHARED_LIBS=OFF \
 -DFAISS_ENABLE_RAFT=ON \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DFAISS_OPT_LEVEL=avx2 \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
 -DCMAKE_C_COMPILER_LAUNCHER=ccache \
 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
 -B build .

make -C build -j12
