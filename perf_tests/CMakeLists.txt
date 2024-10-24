# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every LINEWRAP
project(faiss_perf_tests)
set(BENCHMARK_ENABLE_TESTING OFF)

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
   ${PROJECT_SOURCE_DIR}/../..)

include(../cmake/link_to_faiss_lib.cmake)

link_to_faiss_lib(faiss_perf_tests_utils)

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
   ${PROJECT_SOURCE_DIR}/../..)

endforeach()
