# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(CUDAToolkit REQUIRED)

add_executable(demo_multi_sequence EXCLUDE_FROM_ALL demo_multi_sequence.cpp)
target_link_libraries(demo_multi_sequence PRIVATE faiss CUDA::cudart OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED)
# This has a conditional, because it wan't found in a CI/CD runner
if(MPI_FOUND)
    find_package(OpenMP REQUIRED)
    add_executable(demo_imipq_gpu_sift_m_mpi EXCLUDE_FROM_ALL demo_imipq_gpu_sift_m_mpi.cpp)
    target_link_libraries(demo_imipq_gpu_sift_m_mpi PRIVATE faiss CUDA::cudart MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()
