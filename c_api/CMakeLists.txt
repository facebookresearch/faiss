# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(faiss_c_library LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/c_api/*.cpp")
add_library(faiss_c STATIC ${SOURCES})
target_link_libraries(faiss_c PRIVATE faiss)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/c_api/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/c_api/
        FILES_MATCHING PATTERN "*.h")

add_executable(example_c example_c.c)
target_link_libraries(example_c PRIVATE faiss_c)
