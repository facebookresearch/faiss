# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_TEST_SRC
  test_binary_flat.cpp
  test_dealloc_invlists.cpp
  test_ivfpq_codec.cpp
  test_ivfpq_indexing.cpp
  test_lowlevel_ivf.cpp
  test_ivf_index.cpp
  test_merge.cpp
  test_omp_threads.cpp
  test_ondisk_ivf.cpp
  test_pairs_decoding.cpp
  test_params_override.cpp
  test_pq_encoding.cpp
  test_sliding_ivf.cpp
  test_threaded_index.cpp
  test_transfer_invlists.cpp
  test_mem_leak.cpp
  test_cppcontrib_sa_decode.cpp
  test_cppcontrib_uintreader.cpp
  test_simdlib.cpp
  test_approx_topk.cpp
  test_RCQ_cropping.cpp
  test_distances_simd.cpp
  test_heap.cpp
  test_code_distance.cpp
  test_hnsw.cpp
  test_partitioning.cpp
  test_fastscan_perf.cpp
)

add_executable(faiss_test ${FAISS_TEST_SRC})

if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  target_link_libraries(faiss_test PRIVATE faiss)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  if(NOT WIN32)
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
  else()
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  endif()
  target_link_libraries(faiss_test PRIVATE faiss_avx2)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx512")
  if(NOT WIN32)
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
  else()
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  endif()
  target_link_libraries(faiss_test PRIVATE faiss_avx512)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.12.1.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

target_link_libraries(faiss_test PRIVATE
  OpenMP::OpenMP_CXX
  gtest_main
  $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_test)
