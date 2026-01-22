cmake_minimum_required(VERSION 3.24.0)
project(faiss-wasm VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

# Emscripten-specific settings
set(CMAKE_EXECUTABLE_SUFFIX ".js")

# Include FAISS source
add_subdirectory(../faiss faiss_build)
target_include_directories(faiss PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stubs>)

# Disable features we don't need for WASM
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)

# Define the WASM target
add_executable(faiss-wasm src/bindings.cpp src/blas_stubs.cpp)

# Link against FAISS
target_link_libraries(faiss-wasm PRIVATE faiss)

# Emscripten compiler flags
target_compile_options(faiss-wasm PRIVATE -O3)
target_link_options(faiss-wasm PRIVATE
    -O3
    --bind
    -sALLOW_MEMORY_GROWTH=1
    -sMODULARIZE=1
    -sEXPORT_NAME=FaissModule
    -sWASM=1
    -sNO_DISABLE_EXCEPTION_CATCHING
)
