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

set(CMAKE_CXX_STANDARD 17)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swigfaiss source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
  if(FAISS_ENABLE_GPU)
    set_source_files_properties(${source} PROPERTIES
      COMPILE_DEFINITIONS GPU_WRAPPER
    )
    if (FAISS_ENABLE_ROCM)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_ROCM
      )
    endif()
    if (FAISS_ENABLE_RAFT)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_RAFT
      )
    endif()
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx2.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_sve.swig COPYONLY)

configure_swigfaiss(swigfaiss.swig)
configure_swigfaiss(swigfaiss_avx2.swig)
configure_swigfaiss(swigfaiss_avx512.swig)
configure_swigfaiss(swigfaiss_sve.swig)

if(TARGET faiss)
  # Manually add headers as extra dependencies of swigfaiss.
  set(SWIG_MODULE_swigfaiss_EXTRA_DEPS)
  foreach(h ${FAISS_HEADERS})
    list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
  endforeach()
  if(FAISS_ENABLE_ROCM)
    foreach(h ${FAISS_GPU_HEADERS})
      list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
    endforeach()
  else()
    foreach(h ${FAISS_GPU_HEADERS})
      list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
      list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    endforeach()
  endif()
else()
  find_package(faiss REQUIRED)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "AIX")
swig_add_library(swigfaiss
  TYPE MODULE
  LANGUAGE python
  SOURCES swigfaiss.swig
)
else ()
swig_add_library(swigfaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss.swig
)
endif()

set_property(TARGET swigfaiss PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

set_property(SOURCE swigfaiss_avx2.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_avx2)
swig_add_library(swigfaiss_avx2
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_avx2.swig
)
set_property(TARGET swigfaiss_avx2 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(swigfaiss_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigfaiss_avx512.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_avx512)
swig_add_library(swigfaiss_avx512
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_avx512.swig
)
set_property(TARGET swigfaiss_avx512 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(swigfaiss_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigfaiss_sve.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_sve)
swig_add_library(swigfaiss_sve
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_sve.swig
)
set_property(TARGET swigfaiss_sve PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(swigfaiss_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swigfaiss PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_avx2 PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_avx512 PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_sve PROPERTIES SUFFIX .so)
else()
  # we need bigobj for the swig wrapper
  target_compile_options(swigfaiss PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx2 PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx512 PRIVATE /bigobj)
  target_compile_options(swigfaiss_sve PRIVATE /bigobj)
endif()

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    target_link_libraries(swigfaiss PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx2 PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx512 PRIVATE hip::host)
  else()
    find_package(CUDAToolkit REQUIRED)
    if(FAISS_ENABLE_RAFT)
      find_package(raft COMPONENTS compiled distributed)
    endif()
    target_link_libraries(swigfaiss PRIVATE CUDA::cudart
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
    target_link_libraries(swigfaiss_avx2 PRIVATE CUDA::cudart
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
    target_link_libraries(swigfaiss_avx512 PRIVATE CUDA::cudart
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
    target_link_libraries(swigfaiss_sve PRIVATE CUDA::cudart
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
  endif()
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swigfaiss PRIVATE
  faiss
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx2 PRIVATE
  faiss_avx2
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx512 PRIVATE
  faiss_avx512
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_sve PRIVATE
  faiss_sve
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(swigfaiss PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_sve PRIVATE ${PROJECT_SOURCE_DIR}/../..)

find_package(Python REQUIRED
  COMPONENTS Development NumPy
)

add_library(faiss_python_callbacks EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET faiss_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

if (${CMAKE_SYSTEM_NAME} MATCHES "AIX")
target_link_libraries(faiss_python_callbacks PRIVATE faiss)
endif()

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(faiss_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swigfaiss PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_avx2 PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_avx512 PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_sve PRIVATE faiss_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
configure_file(class_wrappers.py class_wrappers.py COPYONLY)
configure_file(gpu_wrappers.py gpu_wrappers.py COPYONLY)
configure_file(extra_wrappers.py extra_wrappers.py COPYONLY)
configure_file(array_conversions.py array_conversions.py COPYONLY)

file(GLOB files "${PROJECT_SOURCE_DIR}/../../contrib/*.py")
file(COPY ${files} DESTINATION contrib/)
