#!/bin/bash 
MKL_PATH=/home/dq55/intel/oneapi/mkl/latest/lib

rm -rf build
mkdir -p build
cmake -B build \
	-DFAISS_ENABLE_PYTHON=ON \
	-DPYTHON_INCLUDE_DIRS=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
	-DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
	-DBUILD_SHARED_LIBS=OFF \
	-DFAISS_ENABLE_GPU=OFF \
	-DFAISS_ENABLE_RAFT=OFF \
	-DFAISS_ENABLE_C_API=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DBLA_VENDOR=Intel10_64lp \
	"-DMKL_LIBRARIES=-Wl,--start-group;${MKL_PATH}/libmkl_intel_lp64.a;${MKL_PATH}/libmkl_gnu_thread.a;${MKL_PATH}/libmkl_core.a;-Wl,--end-group" 

make -C build -j faiss
make -C build -j swigfaiss

(cd build/faiss/python && python setup.py install)
