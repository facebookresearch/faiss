# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(faiss_benchmarks)

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_INSTALL_DOCS OFF)

include(FetchContent)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main)
FetchContent_MakeAvailable(googlebenchmark)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(gflags REQUIRED)

include(../../cmake/link_to_faiss_lib.cmake)

# Kernel and codec microbenchmarks
set(BENCHMARKS_KERNEL_SRC
  bench_kern_distances.cpp
  bench_kern_extra_distances.cpp
  bench_kern_hamming.cpp
  bench_kern_heap.cpp
  bench_kern_partitioning.cpp
  bench_kern_sorting.cpp
  bench_kern_fp16.cpp
  bench_kern_fwht.cpp
  bench_kern_rabitq.cpp
  bench_kern_visited_table.cpp
  bench_kern_result_handlers.cpp
  bench_codec_pq.cpp
  bench_codec_pq_adc.cpp
  bench_codec_pq_fastscan.cpp
  bench_codec_sq.cpp
  bench_codec_rq.cpp
  bench_codec_kmeans.cpp
  bench_codec_sa_decode.cpp
)

# Index-level benchmarks
set(BENCHMARKS_INDEX_SRC
  bench_index_flat.cpp
  bench_index_flat_quantized.cpp
  bench_index_ivf.cpp
  bench_index_ivf_fastscan.cpp
  bench_index_ivf_big_batch.cpp
  bench_index_ivf_parallel_mode.cpp
  bench_index_ivf_factory.cpp
  bench_index_ivf_selector.cpp
  bench_index_ivfpq_add.cpp
  bench_index_rabitq.cpp
  bench_index_graph.cpp
  bench_index_binary.cpp
  bench_index_composite.cpp
  bench_index_io.cpp
  bench_index_panorama.cpp
  bench_index_pq_polysemous.cpp
  bench_index_rcq.cpp
)

set(ALL_BENCHMARKS_SRC ${BENCHMARKS_KERNEL_SRC} ${BENCHMARKS_INDEX_SRC})

# The AVX2 SIMD template specializations (e.g. faiss::rabitq::
# rearrange_bit_planes<SIMDLevel::AVX2>) are only compiled into the faiss
# library variant linked by link_to_faiss_lib() for these opt levels; at
# "generic"/"sse" the plain faiss library lacks them. Benches that reference
# such symbols directly gate that code on BENCH_FAISS_HAS_AVX2 so the suite
# builds at every opt level.
if(FAISS_OPT_LEVEL STREQUAL "avx2" OR FAISS_OPT_LEVEL STREQUAL "avx512"
    OR FAISS_OPT_LEVEL STREQUAL "avx512_spr" OR FAISS_OPT_LEVEL STREQUAL "dd")
  set(BENCH_FAISS_HAS_AVX2 TRUE)
else()
  set(BENCH_FAISS_HAS_AVX2 FALSE)
endif()

foreach(bench ${ALL_BENCHMARKS_SRC})
  get_filename_component(bench_exec ${bench} NAME_WE)
  add_executable(${bench_exec} ${bench})
  link_to_faiss_lib(${bench_exec})
  target_link_libraries(${bench_exec} PRIVATE OpenMP::OpenMP_CXX benchmark::benchmark gflags)
  target_include_directories(${bench_exec} PRIVATE ${PROJECT_SOURCE_DIR}/..)
  if(BENCH_FAISS_HAS_AVX2)
    target_compile_definitions(${bench_exec} PRIVATE BENCH_FAISS_HAS_AVX2)
  endif()
endforeach()

# Meta-target to build all benchmarks
add_custom_target(benchmarks)
foreach(bench ${ALL_BENCHMARKS_SRC})
  get_filename_component(bench_exec ${bench} NAME_WE)
  add_dependencies(benchmarks ${bench_exec})
endforeach()

