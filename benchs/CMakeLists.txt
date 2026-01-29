# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(bench_ivf_selector EXCLUDE_FROM_ALL bench_ivf_selector.cpp)
target_link_libraries(bench_ivf_selector PRIVATE faiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(bench_result_handler_overhead EXCLUDE_FROM_ALL
  bench_result_handler_overhead.cpp)
target_link_libraries(bench_result_handler_overhead PRIVATE faiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
