# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Metal GPU backend for Apple Silicon.
# Only included when FAISS_ENABLE_METAL=ON and platform is Apple.

set(FAISS_METAL_SRC
  MetalResources.mm
  MetalIndex.mm
  MetalFlatKernels.mm
  MetalIndexFlat.mm
  StandardMetalResources.mm
  MetalCloner.mm
)

add_library(faiss_metal STATIC ${FAISS_METAL_SRC})

target_link_libraries(faiss_metal
  PUBLIC
    faiss
  PRIVATE
    "-framework Metal"
    "-framework MetalKit"
    "-framework MetalPerformanceShaders"
    "-framework Foundation"
)

target_include_directories(faiss_metal
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
  PRIVATE
    ${PROJECT_SOURCE_DIR}
)

target_compile_definitions(faiss_metal PRIVATE FAISS_METAL_ENABLED=1)

set_target_properties(faiss_metal PROPERTIES
  OBJCXX_STANDARD 17
  OBJCXX_STANDARD_REQUIRED ON
)
