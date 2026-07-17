# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# GPU benchmark suite (Google Benchmark). Built only when faiss is configured
# with GPU support (FAISS_ENABLE_GPU=ON) in addition to FAISS_BUILD_BENCHMARKS.
# The benches call the host-side GPU index API only (no device kernels), so
# the sources are plain C++; they still need the CUDA runtime and, when
# enabled, cuVS.

if(NOT FAISS_ENABLE_GPU)
  message(STATUS
    "benchs/gpu: FAISS_ENABLE_GPU=OFF, skipping the GPU benchmark suite")
  return()
endif()

# The googlebenchmark target is FetchContent'd (globally) by benchs/cpp, which
# is added before this directory. find_package results and the
# link_to_faiss_lib function, however, are directory-scoped, so re-resolve
# them here.
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(gflags REQUIRED)
include(../../cmake/link_to_faiss_lib.cmake)

set(GPU_BENCHMARKS_SRC
  bench_gpu_index_flat_ivfpq.cpp
  bench_gpu_index_ivf_1bn.cpp
  bench_gpu_index_ivf_hybrid.cpp
  bench_gpu_index_ivfflat_cuvs.cpp
  bench_gpu_index_ivfpq_cuvs.cpp
)

if(FAISS_ENABLE_ROCM)
  find_package(HIP REQUIRED)
else()
  find_package(CUDAToolkit REQUIRED)
endif()

foreach(bench ${GPU_BENCHMARKS_SRC})
  get_filename_component(bench_exec ${bench} NAME_WE)
  add_executable(${bench_exec} ${bench})
  link_to_faiss_lib(${bench_exec})
  # Reach the shared dataset/CLI helpers in benchs/cpp and this directory.
  target_include_directories(${bench_exec} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)
  target_link_libraries(${bench_exec}
    PRIVATE OpenMP::OpenMP_CXX benchmark::benchmark gflags)
  if(FAISS_ENABLE_ROCM)
    target_link_libraries(${bench_exec} PRIVATE hip::host)
  else()
    target_link_libraries(${bench_exec} PRIVATE CUDA::cudart)
  endif()
  if(FAISS_ENABLE_CUVS)
    target_link_libraries(${bench_exec} PRIVATE cuvs::cuvs)
    # Exposes the use_cuvs cloner/config defaults and gates the cuVS-only
    # code paths inside the benches (see USE_NVIDIA_CUVS guards).
    target_compile_definitions(${bench_exec} PRIVATE USE_NVIDIA_CUVS=1)
  endif()
endforeach()

# Fold the GPU benches into the top-level `benchmarks` meta-target defined by
# benchs/cpp so `cmake --build . --target benchmarks` builds them too.
foreach(bench ${GPU_BENCHMARKS_SRC})
  get_filename_component(bench_exec ${bench} NAME_WE)
  add_dependencies(benchmarks ${bench_exec})
endforeach()
