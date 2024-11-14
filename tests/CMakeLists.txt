# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
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
  test_disable_pq_sdc_tables.cpp
  test_common_ivf_empty_index.cpp
  test_callback.cpp
  test_utils.cpp
)

add_executable(faiss_test ${FAISS_TEST_SRC})

include(../cmake/link_to_faiss_lib.cmake)

link_to_faiss_lib(faiss_test)

if (FAISS_ENABLE_PYTHON)
  target_link_libraries(faiss_test PUBLIC faiss_example_external_module)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850 # release-1.12.1
  OVERRIDE_FIND_PACKAGE)
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

if(NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
   AND NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/GTestConfig.cmake)
  file(
    WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
    [=[
include(CMakeFindDependencyMacro)
find_dependency(googletest)
if(NOT TARGET GTest::GTest)
  add_library(GTest::GTest INTERFACE IMPORTED)
  target_link_libraries(GTest::GTest INTERFACE GTest::gtest)
endif()
if(NOT TARGET GTest::Main)
  add_library(GTest::Main INTERFACE IMPORTED)
  target_link_libraries(GTest::Main INTERFACE GTest::gtest_main)
endif()
]=])
endif()

find_package(OpenMP REQUIRED)
find_package(GTest CONFIG REQUIRED)

target_link_libraries(faiss_test PRIVATE
  OpenMP::OpenMP_CXX
  GTest::gtest_main
  $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>
  $<$<BOOL:${FAISS_ENABLE_ROCM}>:hip::host>
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_test)
