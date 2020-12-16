# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(CUDAToolkit REQUIRED)

# Defines `gtest_discover_tests()`.
include(GoogleTest)

add_library(faiss_gpu_test_helper TestUtils.cpp)
target_link_libraries(faiss_gpu_test_helper PUBLIC faiss gtest CUDA::cudart)

macro(faiss_gpu_test file)
  get_filename_component(test_name ${file} NAME_WE)
  add_executable(${test_name} ${file})
  target_link_libraries(${test_name} PRIVATE faiss_gpu_test_helper)
  gtest_discover_tests(${test_name})
endmacro()

faiss_gpu_test(TestCodePacking.cpp)
faiss_gpu_test(TestGpuIndexFlat.cpp)
faiss_gpu_test(TestGpuIndexIVFFlat.cpp)
faiss_gpu_test(TestGpuIndexBinaryFlat.cpp)
faiss_gpu_test(TestGpuMemoryException.cpp)
faiss_gpu_test(TestGpuIndexIVFPQ.cpp)
faiss_gpu_test(TestGpuIndexIVFScalarQuantizer.cpp)
faiss_gpu_test(TestGpuDistance.cu)
faiss_gpu_test(TestGpuSelect.cu)

add_executable(demo_ivfpq_indexing_gpu EXCLUDE_FROM_ALL
  demo_ivfpq_indexing_gpu.cpp)

target_link_libraries(demo_ivfpq_indexing_gpu
  PRIVATE faiss gtest_main CUDA::cudart)
