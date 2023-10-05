#!/bin/bash

# NOTE: This file is temporary for the proof-of-concept branch and will be removed before this PR is merged

BUILD_TYPE=Release
BUILD_DIR=build/

RAFT_REPO_REL=""
EXTRA_CMAKE_ARGS=""
set -e

if [[ ${RAFT_REPO_REL} != "" ]]; then
  RAFT_REPO_PATH="`readlink -f \"${RAFT_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_raft_SOURCE=${RAFT_REPO_PATH}"
fi

if [ "$1" == "clean" ]; then
  rm -rf build
  rm -rf .cache
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

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake \
 -DFAISS_ENABLE_GPU=ON \
 -DFAISS_ENABLE_RAFT=ON \
 -DFAISS_ENABLE_PYTHON=OFF \
 -DBUILD_TESTING=ON \
 -DBUILD_SHARED_LIBS=OFF \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DFAISS_OPT_LEVEL=avx2 \
 -DRAFT_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
 -DCMAKE_C_COMPILER_LAUNCHER=ccache \
 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
 ${EXTRA_CMAKE_ARGS} \
 ../


# make -C build -j12 faiss
cmake  --build . -j12
# make -C build -j12 swigfaiss
# (cd build/faiss/python && python setup.py install)

