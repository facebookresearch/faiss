#!/bin/bash 
MKL_PATH=/home/dq55/intel/oneapi/mkl/latest/lib

rm -rf build
mkdir -p build
cmake -B build \
	-DBUILD_SHARED_LIBS=OFF \
	-DFAISS_ENABLE_GPU=OFF \
	-DFAISS_ENABLE_RAFT=OFF \
	-DFAISS_ENABLE_PYTHON=OFF \
	-DFAISS_ENABLE_C_API=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DBLA_VENDOR=Intel10_64lp \
	"-DMKL_LIBRARIES=-Wl,--start-group;${MKL_PATH}/libmkl_intel_lp64.a;${MKL_PATH}/libmkl_gnu_thread.a;${MKL_PATH}/libmkl_core.a;-Wl,--end-group" 

make -C build -j faiss
