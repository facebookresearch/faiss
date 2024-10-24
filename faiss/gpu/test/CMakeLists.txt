# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Defines `gtest_discover_tests()`.
include(GoogleTest)
add_library(faiss_gpu_test_helper TestUtils.cpp)
if(FAISS_ENABLE_ROCM)
  target_link_libraries(faiss_gpu_test_helper PUBLIC faiss gtest hip::host)
else()
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(faiss_gpu_test_helper PUBLIC faiss gtest CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs> $<$<BOOL:${FAISS_ENABLE_CUVS}>:OpenMP::OpenMP_CXX>)
endif()

macro(faiss_gpu_test file)
  get_filename_component(test_name ${file} NAME_WE)
  add_executable(${test_name} ${file})
  target_link_libraries(${test_name} PRIVATE faiss_gpu_test_helper)
  target_compile_options(${test_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<BOOL:${FAISS_ENABLE_CUVS}>:-Xcompiler=${OpenMP_CXX_FLAGS}>>)
  gtest_discover_tests(${test_name})
endmacro()


faiss_gpu_test(TestCodePacking.cpp)
faiss_gpu_test(TestGpuIndexFlat.cpp)
faiss_gpu_test(TestGpuIndexIVFFlat.cpp)
faiss_gpu_test(TestGpuIndexBinaryFlat.cpp)
faiss_gpu_test(TestGpuMemoryException.cpp)
faiss_gpu_test(TestGpuIndexIVFPQ.cpp)
faiss_gpu_test(TestGpuIndexIVFScalarQuantizer.cpp)
faiss_gpu_test(TestGpuDistance.${GPU_EXT_PREFIX})
faiss_gpu_test(TestGpuSelect.${GPU_EXT_PREFIX})
if(FAISS_ENABLE_CUVS)
  faiss_gpu_test(TestGpuIndexCagra.cu)
endif()

add_executable(demo_ivfpq_indexing_gpu EXCLUDE_FROM_ALL
  demo_ivfpq_indexing_gpu.cpp)

if (FAISS_ENABLE_ROCM)
  target_link_libraries(demo_ivfpq_indexing_gpu
    PRIVATE faiss gtest_main hip::host)
else()
  target_link_libraries(demo_ivfpq_indexing_gpu
    PRIVATE faiss gtest_main CUDA::cudart)
endif()
