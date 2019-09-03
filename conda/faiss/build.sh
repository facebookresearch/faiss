# Build sse version
CXXFLAGS="-msse4" ./configure --without-cuda
make -j $CPU_COUNT
make -C python _swigfaiss_sse4.so
make clean

# Build avx2 version
CXXFLAGS="-mavx2 -mf16c" ./configure --without-cuda
make -j $CPU_COUNT
make -C python _swigfaiss_avx2.so
make clean

# Build vanilla version (no sse, no avx)
./configure --without-cuda
make -j $CPU_COUNT
make -C python _swigfaiss.so

make -C python build

cd python

$PYTHON setup.py install --single-version-externally-managed --record=record.txt
