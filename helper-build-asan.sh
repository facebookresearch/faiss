#!/bin/bash
# Build Faiss with AddressSanitizer (ASan) enabled
# Exit immediately if any command fails
set -e

# Configure with ASan flags
echo "╔══════════════════════════════════════════════════════╗"
echo "║        Configuring Faiss with AddressSanitizer...    ║"
echo "╚══════════════════════════════════════════════════════╝"
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

# Set ASan environment variables for Python installation
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:symbolize=1:print_stacktrace=1"
export MSAN_OPTIONS="print_stats=1"

cd build/faiss/python && python3 setup.py install --user && cd ../../..

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Faiss with AddressSanitizer is ready! Remember to   ║"
echo "║  set ASAN_OPTIONS when running your programs:        ║"
echo "║  export ASAN_OPTIONS=\"detect_leaks=1:abort_on_error=1\" ║"
echo "╚══════════════════════════════════════════════════════╝"

# run append LD_PRELOAD=$(gcc -print-file-name=libasan.so)