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
  test_merge.cpp
  test_omp_threads.cpp
  test_ondisk_ivf.cpp
  test_pairs_decoding.cpp
  test_params_override.cpp
  test_pq_encoding.cpp
  test_sliding_ivf.cpp
  test_threaded_index.cpp
  test_transfer_invlists.cpp
)

add_executable(faiss_test ${FAISS_TEST_SRC})

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  target_link_libraries(faiss_test PRIVATE faiss_avx2)
else()
  target_link_libraries(faiss_test PRIVATE faiss)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

target_link_libraries(faiss_test PRIVATE
  OpenMP::OpenMP_CXX
  gtest_main
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_test)
