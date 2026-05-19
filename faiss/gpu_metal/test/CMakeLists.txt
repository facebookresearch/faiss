# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Metal backend tests. Only included when BUILD_TESTING and FAISS_ENABLE_METAL.

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(TestMetalIndexFlat TestMetalIndexFlat.mm)
target_link_libraries(TestMetalIndexFlat PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(TestMetalIndexFlat PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(TestMetalIndexFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(TestMetalIndexFlat)
