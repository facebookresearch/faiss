# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

if [ $1 = "cpu" ]; then GPU=OFF; else GPU=ON; fi


# Build avx2 version.
cmake -B _build_python_${PY_VER}_avx2 \
      -Dfaiss_ROOT=_libfaiss_avx2_stage/ \
      -DFAISS_ENABLE_GPU=$GPU \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      faiss/python

cmake --build _build_python_${PY_VER}_avx2 -j $CPU_COUNT


# Build vanilla version (no avx).
cmake -B _build_python_${PY_VER} \
      -Dfaiss_ROOT=_libfaiss_stage/ \
      -DFAISS_ENABLE_GPU=$GPU \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      faiss/python

cmake --build _build_python_${PY_VER} -j $CPU_COUNT


# Build actual python module.
cp _build_python_${PY_VER}_avx2/swigfaiss.py _build_python_${PY_VER}/swigfaiss_avx2.py
cp _build_python_${PY_VER}_avx2/_swigfaiss.so _build_python_${PY_VER}/_swigfaiss_avx2.so
cd _build_python_${PY_VER}/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX
