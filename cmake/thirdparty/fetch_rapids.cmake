# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
set(RAPIDS_VERSION "24.08")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/FAISS_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/FAISS_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/FAISS_RAPIDS.cmake)
