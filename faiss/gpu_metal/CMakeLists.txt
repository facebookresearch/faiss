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
  MetalKernels.mm
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

# Pre-compile Metal shaders: .metal -> .air -> .metallib
find_program(XCRUN xcrun REQUIRED)
set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/MetalDistance.metal)
set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/MetalDistance.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/MetalDistance.metallib)

add_custom_command(
  OUTPUT ${METAL_AIR}
  COMMAND ${XCRUN} -sdk macosx metal -c ${METAL_SOURCE} -o ${METAL_AIR}
  DEPENDS ${METAL_SOURCE}
  COMMENT "Compiling MetalDistance.metal -> .air"
)

add_custom_command(
  OUTPUT ${METAL_LIB}
  COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
  DEPENDS ${METAL_AIR}
  COMMENT "Linking MetalDistance.air -> .metallib"
)

add_custom_target(metal_shaders DEPENDS ${METAL_LIB})
add_dependencies(faiss_metal metal_shaders)

target_compile_definitions(faiss_metal PRIVATE
  FAISS_METALLIB_PATH="${METAL_LIB}"
)
