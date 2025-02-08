@REM Copyright (c) Meta Platforms, Inc. and affiliates.
@REM
@REM This source code is licensed under the MIT license found in the
@REM LICENSE file in the root directory of this source tree.

:: Copyright (c) Facebook, Inc. and its affiliates.
::
:: This source code is licensed under the MIT license found in the
:: LICENSE file in the root directory of this source tree.

:: Build libfaiss.so.
cmake -B _build ^
      -T v141 ^
      -A x64 ^
      -G "Visual Studio 16 2019" ^
      -DBUILD_SHARED_LIBS=ON ^
      -DBUILD_TESTING=OFF ^
      -DFAISS_OPT_LEVEL=avx512 ^
      -DFAISS_ENABLE_GPU=ON ^
      -DFAISS_ENABLE_CUVS=OFF ^
      -DCMAKE_CUDA_ARCHITECTURES="%CUDA_ARCHS%" ^
      -DFAISS_ENABLE_PYTHON=OFF ^
      -DBLA_VENDOR=Intel10_64lp ^
      -DCMAKE_INSTALL_LIBDIR=lib ^
      -DCMAKE_BUILD_TYPE=Release ^
      .
if %errorlevel% neq 0 exit /b %errorlevel%


make -C _build -j %CPU_COUNT% faiss faiss_avx2 faiss_avx512

cmake --install _build --prefix $PREFIX
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --install _build --prefix _libfaiss_stage/
if %errorlevel% neq 0 exit /b %errorlevel%
