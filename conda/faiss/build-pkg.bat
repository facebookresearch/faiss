:: Copyright (c) Facebook, Inc. and its affiliates.
::
:: This source code is licensed under the MIT license found in the
:: LICENSE file in the root directory of this source tree.

:: Build vanilla version (no avx).
cmake -B _build_python_%PY_VER% ^
      -T v141 ^
      -A x64 ^
      -G "Visual Studio 16 2019" ^
      -DFAISS_ENABLE_GPU=OFF ^
      -DPython_EXECUTABLE=%PYTHON% ^
      faiss/python
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build _build_python_%PY_VER% --config Release -j %CPU_COUNT%
if %errorlevel% neq 0 exit /b %errorlevel%


:: Build actual python module.
cd _build_python_%PY_VER%/
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt --prefix=%PREFIX%
if %errorlevel% neq 0 exit /b %errorlevel%
