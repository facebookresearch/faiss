#!/bin/bash
# Build Faiss with optimized AVX512.
# Exit immediately if any command fails
set -e

echo "╔═════════════════════════════════╗"
echo "║       Configuring Faiss ...     ║"
echo "╚═════════════════════════════════╝"
cmake -B build . \
    -DFAISS_OPT_LEVEL=avx512 \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address"

# Build faiss
echo "╔══════════════════════════════════════════════════════╗"
echo "║           Building Faiss core library...             ║"
echo "╚══════════════════════════════════════════════════════╝"
make -C build -j faiss_avx512

echo "╔══════════════════════════════════════════════════════╗"
echo "║         Faiss core library build complete!           ║"
echo "║         Building SWIG Python bindings...             ║"
echo "╚══════════════════════════════════════════════════════╝"
make -C build -j swigfaiss_avx512

echo "╔══════════════════════════════════════════════════════╗"
echo "║        SWIG Python bindings build complete!          ║"
echo "║            Installing Python package...              ║"
echo "╚══════════════════════════════════════════════════════╝"
cd build/faiss/python && python3 setup.py install --user && cd ../../..

echo "╔══════════════════════════════════════════════════════╗"
echo "║    All done! Faiss with Python bindings is ready!    ║"
echo "╚══════════════════════════════════════════════════════╝" 
