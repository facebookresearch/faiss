# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set(RAPIDS_VERSION "23.02")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/FAISS_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/FAISS_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/FAISS_RAPIDS.cmake)
