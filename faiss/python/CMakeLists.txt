# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyfaiss
  DESCRIPTION "Python bindings for faiss."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

set_source_files_properties(swigfaiss.swig PROPERTIES
  CPLUSPLUS ON
  USE_TARGET_INCLUDE_DIRECTORIES TRUE
)

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_source_files_properties(swigfaiss.swig PROPERTIES
    SWIG_FLAGS -DSWIGWORDSIZE64
  )
endif()

swig_add_library(swigfaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss.swig
)

# NOTE: Python does not recognize the dylib extension.
set_target_properties(swigfaiss PROPERTIES SUFFIX .so)

if(FAISS_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(swigfaiss PRIVATE CUDA::cudart)
  set_source_files_properties(swigfaiss.swig PROPERTIES
    COMPILE_DEFINITIONS GPU_WRAPPER
  )
endif()

if(NOT TARGET faiss)
  find_package(faiss REQUIRED)
endif()

target_link_libraries(swigfaiss PRIVATE faiss Python::Module Python::NumPy)
# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(swigfaiss PRIVATE ${PROJECT_SOURCE_DIR}/../../)

find_package(Python REQUIRED
  COMPONENTS Development NumPy
)

add_library(faiss_python_callbacks EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET faiss_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(faiss_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../../)
target_include_directories(faiss_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(faiss_python_callbacks PRIVATE faiss)

target_link_libraries(swigfaiss PRIVATE faiss faiss_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
