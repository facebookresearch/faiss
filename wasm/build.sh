#!/bin/bash
set -e

cd "$(dirname "$0")"
mkdir -p build
cd build

emcmake cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_OPENMP=OFF -DFAISS_ENABLE_BLAS=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_EXTRAS=OFF
emmake make VERBOSE=1

# Copy output to demo directory
cp faiss-wasm.js ../demo/
cp faiss-wasm.wasm ../demo/
