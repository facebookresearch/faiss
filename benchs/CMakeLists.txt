# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(bench_ivf_selector EXCLUDE_FROM_ALL bench_ivf_selector.cpp)
target_link_libraries(bench_ivf_selector PRIVATE faiss_avx512 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bench_ivf_selector PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)

add_executable(bench_result_handler_overhead EXCLUDE_FROM_ALL
  bench_result_handler_overhead.cpp)
target_link_libraries(bench_result_handler_overhead PRIVATE faiss_avx512 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bench_result_handler_overhead PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
