# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Metal backend tests. Only included when BUILD_TESTING and FAISS_ENABLE_METAL.

find_package(GTest CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
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

add_executable(TestMetalIndexIVFFlat TestMetalIndexIVFFlat.mm)
target_link_libraries(TestMetalIndexIVFFlat PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(TestMetalIndexIVFFlat PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(TestMetalIndexIVFFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(TestMetalIndexIVFFlat)

add_executable(TestMetalIndexIVFPQ TestMetalIndexIVFPQ.mm)
target_link_libraries(TestMetalIndexIVFPQ PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(TestMetalIndexIVFPQ PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
set_target_properties(TestMetalIndexIVFPQ PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
gtest_discover_tests(TestMetalIndexIVFPQ)

add_executable(TestMetalIndexBinaryFlat TestMetalIndexBinaryFlat.mm)
target_link_libraries(TestMetalIndexBinaryFlat PRIVATE
    faiss_metal
    faiss
    GTest::gtest_main
)
target_include_directories(TestMetalIndexBinaryFlat PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(TestMetalIndexBinaryFlat PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
target_compile_options(TestMetalIndexBinaryFlat PRIVATE -fobjc-arc)
gtest_discover_tests(TestMetalIndexBinaryFlat)

add_executable(TestMetalBinaryFlatFailures TestMetalBinaryFlatFailures.mm)
target_link_libraries(TestMetalBinaryFlatFailures PRIVATE faiss_metal faiss)
target_include_directories(TestMetalBinaryFlatFailures PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(TestMetalBinaryFlatFailures PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)
target_compile_options(TestMetalBinaryFlatFailures PRIVATE -fobjc-arc)
target_compile_definitions(TestMetalBinaryFlatFailures PRIVATE
    FAISS_TEST_METALLIB_PATH="${PROJECT_BINARY_DIR}/faiss/gpu_metal/MetalDistance.metallib"
)
foreach(fault function pipeline growth scratch)
    add_test(NAME TestMetalIndexBinaryFlat.Reject_${fault}
        COMMAND TestMetalBinaryFlatFailures ${fault})
    set_tests_properties(TestMetalIndexBinaryFlat.Reject_${fault} PROPERTIES
        ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1"
        SKIP_RETURN_CODE 77
    )
endforeach()

add_executable(TestMetalIndexIVFSQ TestMetalIndexIVFSQ.mm)
target_link_libraries(TestMetalIndexIVFSQ PRIVATE faiss_metal faiss GTest::gtest_main)
target_include_directories(TestMetalIndexIVFSQ PRIVATE ${PROJECT_SOURCE_DIR} ${OpenMP_CXX_INCLUDE_DIRS})
set_target_properties(TestMetalIndexIVFSQ PROPERTIES
    OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
target_compile_options(TestMetalIndexIVFSQ PRIVATE -fobjc-arc)
gtest_discover_tests(TestMetalIndexIVFSQ)

add_executable(TestMetalIVFSQFailures TestMetalIVFSQFailures.mm)
target_link_libraries(TestMetalIVFSQFailures PRIVATE faiss_metal faiss)
target_include_directories(TestMetalIVFSQFailures PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(TestMetalIVFSQFailures PROPERTIES
    OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
target_compile_options(TestMetalIVFSQFailures PRIVATE -fobjc-arc)
target_compile_definitions(TestMetalIVFSQFailures PRIVATE
    FAISS_TEST_METALLIB_PATH="${PROJECT_BINARY_DIR}/faiss/gpu_metal/MetalDistance.metallib")
foreach(fault pipeline scratch upload copy growth codec-cache)
    add_test(NAME TestMetalIndexIVFSQ.Reject_${fault}
        COMMAND TestMetalIVFSQFailures ${fault})
    set_tests_properties(TestMetalIndexIVFSQ.Reject_${fault} PROPERTIES
        ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1"
        SKIP_RETURN_CODE 77)
endforeach()
