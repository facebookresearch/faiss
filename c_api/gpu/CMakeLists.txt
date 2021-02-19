# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(faiss_c_gpu_library LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

set(FAISS_C_SRC
  DeviceUtils_c.cpp
  GpuAutoTune_c.cpp
  GpuClonerOptions_c.cpp
  GpuIndex_c.cpp
  GpuResources_c.cpp
  StandardGpuResources_c.cpp
)
target_sources(faiss_c PRIVATE ${FAISS_C_SRC})
target_link_libraries(faiss_c PRIVATE faiss)
install(DIRECTORY ${PROJECT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/
        FILES_MATCHING PATTERN "*.h")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(faiss_c PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_c PRIVATE CUDA::cudart CUDA::cublas)

add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
target_link_libraries(example_gpu_c PRIVATE faiss_c)
