# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(faiss_c_library LANGUAGES C CXX)

target_sources(faiss_c PRIVATE
  DeviceUtils_c.cpp
  GpuAutoTune_c.cpp
  GpuClonerOptions_c.cpp
  GpuIndex_c.cpp
  GpuResources_c.cpp
  StandardGpuResources_c.cpp
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/c_api/gpu/
        FILES_MATCHING PATTERN "*.h")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(faiss_c PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_c PRIVATE CUDA::cudart CUDA::cublas)

add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
target_link_libraries(example_gpu_c PRIVATE faiss_c)
