@REM Copyright (c) Meta Platforms, Inc. and affiliates.
@REM
@REM This source code is licensed under the MIT license found in the
@REM LICENSE file in the root directory of this source tree.

:: Copyright (c) Facebook, Inc. and its affiliates.
::
:: This source code is licensed under the MIT license found in the
:: LICENSE file in the root directory of this source tree.

:: Build vanilla version (no avx).
cmake -B _build_python_%PY_VER% ^
      -T v141 ^
      -A x64 ^
      -G "Visual Studio 16 2019" ^
      -Dfaiss_ROOT=_libfaiss_stage/ ^
      -DFAISS_OPT_LEVEL=avx512 ^
      -DFAISS_ENABLE_GPU=ON ^
      -DFAISS_ENABLE_CUVS=OFF ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DPython_EXECUTABLE=$PYTHON ^
      faiss/python

if %errorlevel% neq 0 exit /b %errorlevel%

make -C _build_python_%PY_VER% -j %CPU_COUNT% swigfaiss swigfaiss_avx2 swigfaiss_avx512

::cmake --build _build_python_%PY_VER% --config Release -j %CPU_COUNT%
if %errorlevel% neq 0 exit /b %errorlevel%


:: Build actual python module.
cd _build_python_%PY_VER%/
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt --prefix=%PREFIX%
if %errorlevel% neq 0 exit /b %errorlevel%
