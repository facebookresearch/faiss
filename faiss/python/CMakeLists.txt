# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyfaiss
  DESCRIPTION "Python bindings for faiss."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

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
  if(FAISS_ENABLE_SVS)
    set_property(SOURCE ${source} APPEND PROPERTY
      SWIG_FLAGS -DFAISS_ENABLE_SVS -DFAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION}
    )
  endif()
  if(FAISS_OPT_LEVEL STREQUAL "dd")
    set_property(SOURCE ${source} APPEND PROPERTY
      SWIG_FLAGS -DFAISS_ENABLE_DD
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
    if (FAISS_ENABLE_CUVS)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_CUVS
      )
    endif()
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx2.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512_spr.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_sve.swig COPYONLY)

configure_swigfaiss(swigfaiss.swig)
configure_swigfaiss(swigfaiss_avx2.swig)
configure_swigfaiss(swigfaiss_avx512.swig)
configure_swigfaiss(swigfaiss_avx512_spr.swig)
configure_swigfaiss(swigfaiss_sve.swig)
configure_swigfaiss(faiss_example_external_module.swig)

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
    list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
  endforeach()
  foreach(h ${FAISS_GPU_HEADERS})
    list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
  endforeach()
else()
  find_package(faiss REQUIRED)
endif()

swig_add_library(swigfaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss.swig
)

if(TARGET swigfaiss AND NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set_target_properties(swigfaiss PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
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
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
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

set_property(SOURCE swigfaiss_avx512_spr.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_avx512_spr)
swig_add_library(swigfaiss_avx512_spr
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_avx512_spr.swig
)
set_property(TARGET swigfaiss_avx512_spr PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(swigfaiss_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
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

set_property(SOURCE faiss_example_external_module.swig
  PROPERTY SWIG_MODULE_NAME faiss_example_external_module)
swig_add_library(faiss_example_external_module
  TYPE SHARED
  LANGUAGE python
  SOURCES faiss_example_external_module.swig
)
set_property(TARGET faiss_example_external_module PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

if(AIX)
  set_target_properties(swigfaiss PROPERTIES SUFFIX .a)
  set_target_properties(faiss_example_external_module PROPERTIES SUFFIX .a)
elseif(WIN32)
  # we need bigobj for the swig wrapper
  target_compile_options(swigfaiss PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx2 PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx512 PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx512_spr PRIVATE /bigobj)
  target_compile_options(swigfaiss_sve PRIVATE /bigobj)
  target_compile_options(faiss_example_external_module PRIVATE /bigobj)
elseif(NOT SKBUILD_SABI_VERSION)
  # NOTE: Python does not recognize the dylib extension.
  # When building abi3, the suffix is set to .abi3.so below.
  set_target_properties(swigfaiss PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_avx2 PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_avx512 PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_avx512_spr PROPERTIES SUFFIX .so)
  set_target_properties(swigfaiss_sve PROPERTIES SUFFIX .so)
  set_target_properties(faiss_example_external_module PROPERTIES SUFFIX .so)
endif()

if(FAISS_ENABLE_SVS)
  target_compile_definitions(swigfaiss PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_avx2 PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_avx512 PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_avx512_spr PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_sve PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(faiss_example_external_module PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
endif()

# Enable DD support in Python bindings when building with Dynamic Dispatch
if(FAISS_OPT_LEVEL STREQUAL "dd")
  # Add compile definition for the generated C++ wrapper
  target_compile_definitions(swigfaiss PRIVATE FAISS_ENABLE_DD)
endif()

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    target_link_libraries(swigfaiss PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx2 PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx512 PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx512_spr PRIVATE hip::host)
    target_link_libraries(faiss_example_external_module PRIVATE hip::host)
  else()
    find_package(CUDAToolkit REQUIRED)
    if(FAISS_ENABLE_CUVS AND NOT TARGET cuvs::cuvs)
      find_package(cuvs)
    endif()
    target_link_libraries(swigfaiss PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_avx2 PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_avx512 PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_avx512_spr PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_sve PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  endif()
endif()

find_package(OpenMP REQUIRED)

# When scikit-build-core requests abi3 (Stable ABI), it passes
# SKBUILD_SABI_VERSION. Use Development.SABIModule to link against the
# stable Python ABI instead of the version-specific one.
if(SKBUILD_SABI_VERSION)
  find_package(Python REQUIRED
    COMPONENTS Development.SABIModule NumPy
  )
else()
  find_package(Python REQUIRED
    COMPONENTS Development.Module NumPy
  )
endif()

# Select the Python module target based on whether abi3 is requested.
if(SKBUILD_SABI_VERSION)
  set(FAISS_PYTHON_MODULE_TARGET Python::SABIModule)
else()
  set(FAISS_PYTHON_MODULE_TARGET Python::Module)
endif()

# When building for abi3, define Py_LIMITED_API and set the .abi3.so suffix
# so the wheel is tagged as abi3-compatible.
if(SKBUILD_SABI_VERSION)
  # 0x030A0000 = Python 3.10
  set(FAISS_PY_LIMITED_API "0x030A0000")
  foreach(tgt swigfaiss swigfaiss_avx2 swigfaiss_avx512
      swigfaiss_avx512_spr swigfaiss_sve faiss_example_external_module)
    if(TARGET ${tgt})
      target_compile_definitions(${tgt}
        PRIVATE Py_LIMITED_API=${FAISS_PY_LIMITED_API})
      if(NOT WIN32)
        set_target_properties(${tgt} PROPERTIES SUFFIX ".abi3.so")
      endif()
    endif()
  endforeach()
endif()

target_link_libraries(swigfaiss PRIVATE
  faiss
  ${FAISS_PYTHON_MODULE_TARGET}
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx2 PRIVATE
  faiss_avx2
  ${FAISS_PYTHON_MODULE_TARGET}
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx512 PRIVATE
  faiss_avx512
  ${FAISS_PYTHON_MODULE_TARGET}
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx512_spr PRIVATE
  faiss_avx512_spr
  ${FAISS_PYTHON_MODULE_TARGET}
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_sve PRIVATE
  faiss_sve
  ${FAISS_PYTHON_MODULE_TARGET}
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(faiss_example_external_module PRIVATE
  ${FAISS_PYTHON_MODULE_TARGET}
  Python::NumPy
  OpenMP::OpenMP_CXX
  swigfaiss
  faiss
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(swigfaiss PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx512_spr PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_sve PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_example_external_module PRIVATE ${PROJECT_SOURCE_DIR}/../..)

add_library(faiss_python_callbacks STATIC EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET faiss_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)
if(SKBUILD_SABI_VERSION)
  target_compile_definitions(faiss_python_callbacks
    PRIVATE Py_LIMITED_API=${FAISS_PY_LIMITED_API})
endif()

if ("${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
target_link_libraries(faiss_python_callbacks PRIVATE faiss)
endif()

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(faiss_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swigfaiss PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_avx2 PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_avx512 PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_avx512_spr PRIVATE faiss_python_callbacks)
target_link_libraries(swigfaiss_sve PRIVATE faiss_python_callbacks)
target_link_libraries(faiss_example_external_module PRIVATE faiss_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
configure_file(class_wrappers.py class_wrappers.py COPYONLY)
configure_file(gpu_wrappers.py gpu_wrappers.py COPYONLY)
configure_file(extra_wrappers.py extra_wrappers.py COPYONLY)
configure_file(array_conversions.py array_conversions.py COPYONLY)

# file(GLOB files "${PROJECT_SOURCE_DIR}/../../contrib/*.py")
file(COPY ${PROJECT_SOURCE_DIR}/../../contrib  DESTINATION .)

# Install targets for scikit-build-core / pip wheel packaging.

# Set RPATH so the SWIG modules find libfaiss in the same directory.
# auditwheel/delocate follow RPATH to locate libraries for bundling.
foreach(tgt swigfaiss swigfaiss_avx2 swigfaiss_avx512
    swigfaiss_avx512_spr swigfaiss_sve)
  if(TARGET ${tgt})
    if(APPLE)
      set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "@loader_path")
    else()
      set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
  endif()
endforeach()

# Install the SWIG extension modules (.so/.pyd).
# In DD mode, swigfaiss_avx2/avx512/avx512_spr/sve are also built and
# loader.py selects the best one at runtime based on CPU capabilities.
foreach(tgt swigfaiss swigfaiss_avx2 swigfaiss_avx512
    swigfaiss_avx512_spr swigfaiss_sve)
  if(TARGET ${tgt})
    get_target_property(_excluded ${tgt} EXCLUDE_FROM_ALL)
    if(NOT _excluded)
      install(TARGETS ${tgt}
        LIBRARY DESTINATION faiss
        RUNTIME DESTINATION faiss
      )
    endif()
  endif()
endforeach()

# Install the core faiss shared libraries into the Python package directory
# so that auditwheel/delocate can find and bundle them with the SWIG modules.
# Guard: when Python bindings are built standalone (e.g. conda), the faiss
# target is an installed import library, not a build target.
foreach(tgt faiss faiss_avx2 faiss_avx512 faiss_avx512_spr faiss_sve)
  if(TARGET ${tgt})
    get_target_property(_imported ${tgt} IMPORTED)
    get_target_property(_excluded ${tgt} EXCLUDE_FROM_ALL)
    if(NOT _imported AND NOT _excluded)
      install(TARGETS ${tgt}
        LIBRARY DESTINATION faiss
        RUNTIME DESTINATION faiss
      )
    endif()
  endif()
endforeach()

# Install SWIG-generated Python wrappers.
# In DD mode, each variant has its own .py wrapper that loader.py imports.
foreach(mod swigfaiss swigfaiss_avx2 swigfaiss_avx512
    swigfaiss_avx512_spr swigfaiss_sve)
  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${mod}.py" OR TARGET ${mod})
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${mod}.py"
      DESTINATION faiss
      OPTIONAL
    )
  endif()
endforeach()

# Install hand-written Python source files
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/loader.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/class_wrappers.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/extra_wrappers.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu_wrappers.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/array_conversions.py"
  DESTINATION faiss
)

# Install type stubs if present
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.pyi")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.pyi" DESTINATION faiss)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/py.typed")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/py.typed" DESTINATION faiss)
endif()

# Install contrib subpackage (excluding Meta-internal *_fb.py files)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../contrib/"
  DESTINATION faiss/contrib
  FILES_MATCHING PATTERN "*.py"
  PATTERN "*_fb.py" EXCLUDE
)

# Install contrib/torch subpackage data files
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../contrib/torch/"
  DESTINATION faiss/contrib/torch
  FILES_MATCHING PATTERN "*.md"
)
