REM Builds FAISS for Windows ARM64
REM 
set ARCH=%1
shift /1
if defined ARCH (echo ARCH=!ARCH!) else (goto :usage)

REM absolute path to OpenBLAS install, e.g. \d\OpenBLAS\build\install
set OPENBLAS=%~f1
shift /1
if defined OPENBLAS (echo OPENBLAS=!OPENBLAS!) else (goto :usage)

goto :main

:usage
@echo Usage: %0 "x64|arm64" "path-to-openblas-build"
exit /b 1

:main
if [%ARCH%] == [x64] ( 
  set EXTRA_ARGS="-DFAISS_OPT_LEVEL=avx2"
)
if [%ARCH%] == [arm64] ( 
  set EXTRA_ARGS="-DFAISS_OPT_LEVEL=generic"
)

mkdir build_%ARCH%
cd build_%ARCH%
rem Testing is off in https://github.com/facebookresearch/faiss/blob/4012a788ee36132fbd4a454addb7fa2bc134d89e/conda/faiss/build-lib.bat#L12
rem Instead, remove offending tests
cmake .. -A %ARCH% ^
rem needs to copy faiss.dll, gtest*.dll to build_%ARCH%\tests\Release to work with tests. 
  -DFAISS_ENABLE_GPU=OFF ^
  -DFAISS_ENABLE_PYTHON=OFF ^
  -DBUILD_SHARED_LIBS=ON ^
  -DBUILD_TESTING=OFF ^
  -DFAISS_ENABLE_C_API=ON ^
  -DBLA_VENDOR=OpenBLAS ^
  -DBLAS_ROOT=%OPENBLAS%\share\cmake\OpenBLAS ^
  -DCMAKE_PREFIX_PATH=%OPENBLAS%\lib -DCMAKE_FIND_DEBUG_MODE=TRUE %EXTRA_ARGS% || exit /b 1
cmake --build . --config Release --target ALL_BUILD