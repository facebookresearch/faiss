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
      -DFAISS_ENABLE_GPU=OFF ^
      -DFAISS_ENABLE_PYTHON=OFF ^
      -DBLA_VENDOR=Intel10_64_dyn ^
      .
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build _build --config Release -j %CPU_COUNT%
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --install _build --config Release --prefix %PREFIX%
if %errorlevel% neq 0 exit /b %errorlevel%
