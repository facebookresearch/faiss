# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

function(find_and_configure_cutlass)
    set(oneValueArgs VERSION REPOSITORY PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # if(RAFT_ENABLE_DIST_DEPENDENCIES OR RAFT_COMPILE_LIBRARIES)
    set(CUTLASS_ENABLE_HEADERS_ONLY
            ON
            CACHE BOOL "Enable only the header library"
            )
    set(CUTLASS_NAMESPACE
            "raft_cutlass"
            CACHE STRING "Top level namespace of CUTLASS"
            )
    set(CUTLASS_ENABLE_CUBLAS
            OFF
            CACHE BOOL "Disable CUTLASS to build with cuBLAS library."
            )

    if (CUDA_STATIC_RUNTIME)
        set(CUDART_LIBRARY "${CUDA_cudart_static_LIBRARY}" CACHE FILEPATH "fixing cutlass cmake code" FORCE)
    endif()

    rapids_cpm_find(
            NvidiaCutlass ${PKG_VERSION}
            GLOBAL_TARGETS nvidia::cutlass::cutlass
            CPM_ARGS
            GIT_REPOSITORY ${PKG_REPOSITORY}
            GIT_TAG ${PKG_PINNED_TAG}
            GIT_SHALLOW TRUE
            OPTIONS "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
    )

    if(TARGET CUTLASS AND NOT TARGET nvidia::cutlass::cutlass)
        add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
    endif()

    if(NvidiaCutlass_ADDED)
        rapids_export(
                BUILD NvidiaCutlass
                EXPORT_SET NvidiaCutlass
                GLOBAL_TARGETS nvidia::cutlass::cutlass
                NAMESPACE nvidia::cutlass::
        )
    endif()
    # endif()

    # We generate the cutlass-config files when we built cutlass locally, so always do
    # `find_dependency`
    rapids_export_package(
            BUILD NvidiaCutlass raft-distance-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
    )
    rapids_export_package(
            INSTALL NvidiaCutlass raft-distance-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
    )
    rapids_export_package(
            BUILD NvidiaCutlass raft-nn-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
    )
    rapids_export_package(
            INSTALL NvidiaCutlass raft-nn-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
    )

    # Tell cmake where it can find the generated NvidiaCutlass-config.cmake we wrote.
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
            INSTALL NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}/../]=] raft-distance-exports
    )
    rapids_export_find_package_root(
            BUILD NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}]=] raft-distance-exports
    )
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
            INSTALL NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}/../]=] raft-nn-exports
    )
    rapids_export_find_package_root(
            BUILD NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}]=] raft-nn-exports
    )
endfunction()

if(NOT RAFT_CUTLASS_GIT_TAG)
    set(RAFT_CUTLASS_GIT_TAG v2.9.1)
endif()

if(NOT RAFT_CUTLASS_GIT_REPOSITORY)
    set(RAFT_CUTLASS_GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git)
endif()

find_and_configure_cutlass(
        VERSION 2.9.1 REPOSITORY ${RAFT_CUTLASS_GIT_REPOSITORY} PINNED_TAG ${RAFT_CUTLASS_GIT_TAG}
)
