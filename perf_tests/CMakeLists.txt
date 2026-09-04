# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every LINEWRAP
project(faiss_perf_tests)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_INSTALL_DOCS OFF)

include(FetchContent)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark
FetchContent_MakeAvailable(
  googlebenchmark)


find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(gflags REQUIRED)

add_library(faiss_perf_tests_utils
  utils.cpp
)
# `#include <faiss/perf_tests/utils.h>` or any other headers
target_include_directories(faiss_perf_tests_utils PRIVATE
   ${PROJECT_SOURCE_DIR}/..)

include(../cmake/link_to_faiss_lib.cmake)

link_to_faiss_lib(faiss_perf_tests_utils)

if(FAISS_OPT_LEVEL STREQUAL "dd" AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
  # The benchmark directly instantiates DD-only AVX2/AVX-512 symbols. Keep it
  # opt-in so generic/avx2 default builds neither link nor run unavailable ISA.
  add_executable(bench_rabitq_simd EXCLUDE_FROM_ALL ../benchs/bench_rabitq_simd.cpp)
  link_to_faiss_lib(bench_rabitq_simd)
  target_link_libraries(bench_rabitq_simd PRIVATE benchmark::benchmark)
endif()

set(FAISS_PERF_TEST_SRC
  bench_no_multithreading_rcq_search.cpp
  bench_scalar_quantizer_accuracy.cpp
  bench_scalar_quantizer_decode.cpp
  bench_scalar_quantizer_distance.cpp
  bench_scalar_quantizer_encode.cpp
)
foreach(bench ${FAISS_PERF_TEST_SRC})
  get_filename_component(bench_exec ${bench} NAME_WE)
  add_executable(${bench_exec} ${bench})
  link_to_faiss_lib(${bench_exec})
  target_link_libraries(${bench_exec} PRIVATE faiss_perf_tests_utils OpenMP::OpenMP_CXX benchmark::benchmark gflags)
  # `#include <faiss/perf_tests/utils.h>` or any other headers
  target_include_directories(${bench_exec} PRIVATE
   ${PROJECT_SOURCE_DIR}/..)

endforeach()
