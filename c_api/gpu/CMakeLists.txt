# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

target_sources(faiss_c PRIVATE
  DeviceUtils_c.cpp
  GpuAutoTune_c.cpp
  GpuClonerOptions_c.cpp
  GpuIndex_c.cpp
  GpuResources_c.cpp
  StandardGpuResources_c.cpp
)

file(GLOB FAISS_C_API_GPU_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
faiss_install_headers("${FAISS_C_API_GPU_HEADERS}" c_api/gpu)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(faiss_c PUBLIC CUDA::cudart CUDA::cublas)

add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
target_link_libraries(example_gpu_c PRIVATE faiss_c)
