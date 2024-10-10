# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
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

function(link_to_faiss_lib target)
  if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "sve")
    target_link_libraries(${target} PRIVATE faiss)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx2")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    target_link_libraries(${target} PRIVATE faiss_avx2)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx512")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE faiss_avx512)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "sve")
    if(NOT WIN32)
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
      endif()
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
      endif()
    else()
      # TODO: support Windows
    endif()
    target_link_libraries(${target} PRIVATE faiss_sve)
  endif()
endfunction()

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
