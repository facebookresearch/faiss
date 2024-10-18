# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from CMake's FindBLAS module.
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMKL
--------

Find Intel MKL library.

Input Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``BLA_STATIC``
  if ``ON`` use static linkage

``BLA_VENDOR``
  If set, checks only the specified vendor, if not set checks all the
  possibilities.  List of vendors valid in this module:

  * ``Intel10_32`` (intel mkl v10 32 bit)
  * ``Intel10_64lp`` (intel mkl v10+ 64 bit, threaded code, lp64 model)
  * ``Intel10_64lp_seq`` (intel mkl v10+ 64 bit, sequential code, lp64 model)
  * ``Intel10_64ilp`` (intel mkl v10+ 64 bit, threaded code, ilp64 model)
  * ``Intel10_64ilp_seq`` (intel mkl v10+ 64 bit, sequential code, ilp64 model)
  * ``Intel10_64_dyn`` (intel mkl v10+ 64 bit, single dynamic library)
  * ``Intel`` (obsolete versions of mkl 32 and 64 bit)


Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``MKL_FOUND``
  library implementing the BLAS interface is found
``MKL_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  to use MKL (may be empty if compiler implicitly links MKL)

.. note::

  C or CXX must be enabled to use Intel Math Kernel Library (MKL).

  For example, to use Intel MKL libraries and/or Intel compiler:

  .. code-block:: cmake

    set(BLA_VENDOR Intel10_64lp)
    find_package(MKL)

Hints
^^^^^

Set the ``MKLROOT`` environment variable to a directory that contains an MKL
installation, or add the directory to the dynamic library loader environment
variable for your platform (``LIB``, ``DYLD_LIBRARY_PATH`` or
``LD_LIBRARY_PATH``).

#]=======================================================================]

include(CheckFunctionExists)
include(CMakePushCheckState)
include(FindPackageHandleStandardArgs)
cmake_push_check_state()
set(CMAKE_REQUIRED_QUIET ${BLAS_FIND_QUIETLY})


set(_blas_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
if(BLA_STATIC)
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # for ubuntu's libblas3gf and liblapack3gf packages
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .so.3gf)
  endif()
endif()

macro(CHECK_BLAS_LIBRARIES LIBRARIES _prefix _name _flags _list _threadlibs _addlibdir _subdirs)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.

  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  # _addlibdir is a list of additional search paths. _subdirs is a list of path
  # suffixes to be used by find_library().

  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)

  set(_extaddlibdir "${_addlibdir}")
  if(WIN32)
    list(APPEND _extaddlibdir ENV LIB)
  elseif(APPLE)
    list(APPEND _extaddlibdir ENV DYLD_LIBRARY_PATH)
  else()
    list(APPEND _extaddlibdir ENV LD_LIBRARY_PATH)
  endif()
  list(APPEND _extaddlibdir "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")

  foreach(_library ${_list})
    if(_library MATCHES "^-Wl,--(start|end)-group$")
      # Respect linker flags like --start/end-group (required by MKL)
      set(${LIBRARIES} ${${LIBRARIES}} "${_library}")
    else()
      set(_combined_name ${_combined_name}_${_library})
      if(NOT "${_threadlibs}" STREQUAL "")
        set(_combined_name ${_combined_name}_threadlibs)
      endif()
      if(_libraries_work)
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS ${_extaddlibdir}
          PATH_SUFFIXES ${_subdirs}
        )
        #message("DEBUG: find_library(${_library}) got ${${_prefix}_${_library}_LIBRARY}")
        mark_as_advanced(${_prefix}_${_library}_LIBRARY)
        set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
        set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      endif()
    endif()
  endforeach()

  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}} ${_threadlibs})
    #message("DEBUG: CMAKE_REQUIRED_LIBRARIES = ${CMAKE_REQUIRED_LIBRARIES}")
    if(CMAKE_Fortran_COMPILER_LOADED)
      check_fortran_function_exists("${_name}" ${_prefix}${_combined_name}_WORKS)
    else()
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif()
    set(CMAKE_REQUIRED_LIBRARIES)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif()

  if(_libraries_work)
    if("${_list}" STREQUAL "")
      set(${LIBRARIES} "${LIBRARIES}-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
    else()
      set(${LIBRARIES} ${${LIBRARIES}} ${_threadlibs})
    endif()
  else()
    set(${LIBRARIES} FALSE)
  endif()
  #message("DEBUG: ${LIBRARIES} = ${${LIBRARIES}}")
endmacro()

set(MKL_LIBRARIES)
if(NOT $ENV{BLA_VENDOR} STREQUAL "")
  set(BLA_VENDOR $ENV{BLA_VENDOR})
else()
  if(NOT BLA_VENDOR)
    set(BLA_VENDOR "All")
  endif()
endif()
if(BLA_VENDOR_THREADING)
  set(BLAS_mkl_THREADING ${BLA_VENDOR_THREADING})
else()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(BLAS_mkl_THREADING "gnu")
  else()
    set(BLAS_mkl_THREADING "intel")
  endif()
endif()

if(CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED)
  # System-specific settings
  if(WIN32)
    if(BLA_STATIC)
      set(BLAS_mkl_DLL_SUFFIX "")
    else()
      set(BLAS_mkl_DLL_SUFFIX "_dll")
    endif()
  else()
    if(BLA_STATIC)
      set(BLAS_mkl_START_GROUP "-Wl,--start-group")
      set(BLAS_mkl_END_GROUP "-Wl,--end-group")
    else()
      set(BLAS_mkl_START_GROUP "")
      set(BLAS_mkl_END_GROUP "")
    endif()
    if(BLAS_mkl_THREADING STREQUAL "gnu")
      set(BLAS_mkl_OMP "gomp")
    else()
      set(BLAS_mkl_OMP "iomp5")
    endif()
    set(BLAS_mkl_LM "-lm")
    set(BLAS_mkl_LDL "-ldl")
  endif()

  if(BLAS_FIND_QUIETLY OR NOT BLAS_FIND_REQUIRED)
    find_package(Threads)
  else()
    find_package(Threads REQUIRED)
  endif()

  set(BLAS_mkl_INTFACE "intel")
  if(BLA_VENDOR MATCHES "_64ilp")
    set(BLAS_mkl_ILP_MODE "ilp64")
  else()
    set(BLAS_mkl_ILP_MODE "lp64")
  endif()

  set(BLAS_SEARCH_LIBS "")

  set(BLAS_mkl_SEARCH_SYMBOL sgemm)
  set(_LIBRARIES MKL_LIBRARIES)
  if(WIN32)
    # Find the main file (32-bit or 64-bit)
    set(BLAS_SEARCH_LIBS_WIN_MAIN "")
    if(BLA_VENDOR STREQUAL "Intel10_32" OR BLA_VENDOR STREQUAL "All")
      list(APPEND BLAS_SEARCH_LIBS_WIN_MAIN
        "mkl_intel_c${BLAS_mkl_DLL_SUFFIX}")
    endif()
    if(BLA_VENDOR MATCHES "^Intel10_64i?lp" OR BLA_VENDOR STREQUAL "All")
      list(APPEND BLAS_SEARCH_LIBS_WIN_MAIN
        "mkl_intel_${BLAS_mkl_ILP_MODE}${BLAS_mkl_DLL_SUFFIX}")
    endif()

    # Add threading/sequential libs
    set(BLAS_SEARCH_LIBS_WIN_THREAD "")
    if(BLA_VENDOR MATCHES "^Intel10_64i?lp$" OR BLA_VENDOR STREQUAL "All")
      # old version
      list(APPEND BLAS_SEARCH_LIBS_WIN_THREAD
        "libguide40 mkl_intel_thread${BLAS_mkl_DLL_SUFFIX}")
      # mkl >= 10.3
      list(APPEND BLAS_SEARCH_LIBS_WIN_THREAD
        "libiomp5md mkl_intel_thread${BLAS_mkl_DLL_SUFFIX}")
    endif()
    if(BLA_VENDOR MATCHES "^Intel10_64i?lp_seq$" OR BLA_VENDOR STREQUAL "All")
      list(APPEND BLAS_SEARCH_LIBS_WIN_THREAD
        "mkl_sequential${BLAS_mkl_DLL_SUFFIX}")
    endif()

    # Cartesian product of the above
    foreach(MAIN ${BLAS_SEARCH_LIBS_WIN_MAIN})
      foreach(THREAD ${BLAS_SEARCH_LIBS_WIN_THREAD})
        list(APPEND BLAS_SEARCH_LIBS
          "${MAIN} ${THREAD} mkl_core${BLAS_mkl_DLL_SUFFIX}")
      endforeach()
    endforeach()
  else()
    if(BLA_VENDOR STREQUAL "Intel10_32" OR BLA_VENDOR STREQUAL "All")
      # old version
      list(APPEND BLAS_SEARCH_LIBS
        "mkl_${BLAS_mkl_INTFACE} mkl_${BLAS_mkl_THREADING}_thread mkl_core guide")

      # mkl >= 10.3
      list(APPEND BLAS_SEARCH_LIBS
        "${BLAS_mkl_START_GROUP} mkl_${BLAS_mkl_INTFACE} mkl_${BLAS_mkl_THREADING}_thread mkl_core ${BLAS_mkl_END_GROUP} ${BLAS_mkl_OMP}")
    endif()
    if(BLA_VENDOR MATCHES "^Intel10_64i?lp$" OR BLA_VENDOR STREQUAL "All")
      # old version
      list(APPEND BLAS_SEARCH_LIBS
        "mkl_${BLAS_mkl_INTFACE}_${BLAS_mkl_ILP_MODE} mkl_${BLAS_mkl_THREADING}_thread mkl_core guide")

      # mkl >= 10.3
      list(APPEND BLAS_SEARCH_LIBS
        "${BLAS_mkl_START_GROUP} mkl_${BLAS_mkl_INTFACE}_${BLAS_mkl_ILP_MODE} mkl_${BLAS_mkl_THREADING}_thread mkl_core ${BLAS_mkl_END_GROUP} ${BLAS_mkl_OMP}")
    endif()
    if(BLA_VENDOR MATCHES "^Intel10_64i?lp_seq$" OR BLA_VENDOR STREQUAL "All")
      list(APPEND BLAS_SEARCH_LIBS
        "${BLAS_mkl_START_GROUP} mkl_${BLAS_mkl_INTFACE}_${BLAS_mkl_ILP_MODE} mkl_sequential mkl_core ${BLAS_mkl_END_GROUP}")
    endif()

    #older vesions of intel mkl libs
    if(BLA_VENDOR STREQUAL "Intel" OR BLA_VENDOR STREQUAL "All")
      list(APPEND BLAS_SEARCH_LIBS
        "mkl")
      list(APPEND BLAS_SEARCH_LIBS
        "mkl_ia32")
      list(APPEND BLAS_SEARCH_LIBS
        "mkl_em64t")
    endif()
  endif()

  if(BLA_VENDOR MATCHES "^Intel10_64_dyn$" OR BLA_VENDOR STREQUAL "All")
    # mkl >= 10.3 with single dynamic library
    list(APPEND BLAS_SEARCH_LIBS
      "mkl_rt")
  endif()

  # MKL uses a multitude of partially platform-specific subdirectories:
  if(BLA_VENDOR STREQUAL "Intel10_32")
    set(BLAS_mkl_ARCH_NAME "ia32")
  else()
    set(BLAS_mkl_ARCH_NAME "intel64")
  endif()
  if(WIN32)
    set(BLAS_mkl_OS_NAME "win")
  elseif(APPLE)
    set(BLAS_mkl_OS_NAME "mac")
  else()
    set(BLAS_mkl_OS_NAME "lin")
  endif()
  if(DEFINED ENV{MKLROOT})
    file(TO_CMAKE_PATH "$ENV{MKLROOT}" BLAS_mkl_MKLROOT)
    # If MKLROOT points to the subdirectory 'mkl', use the parent directory instead
    # so we can better detect other relevant libraries in 'compiler' or 'tbb':
    get_filename_component(BLAS_mkl_MKLROOT_LAST_DIR "${BLAS_mkl_MKLROOT}" NAME)
    if(BLAS_mkl_MKLROOT_LAST_DIR STREQUAL "mkl")
      get_filename_component(BLAS_mkl_MKLROOT "${BLAS_mkl_MKLROOT}" DIRECTORY)
    endif()
  endif()
  set(BLAS_mkl_LIB_PATH_SUFFIXES
    "compiler/lib" "compiler/lib/${BLAS_mkl_ARCH_NAME}_${BLAS_mkl_OS_NAME}"
    "mkl/lib" "mkl/lib/${BLAS_mkl_ARCH_NAME}_${BLAS_mkl_OS_NAME}"
    "lib/${BLAS_mkl_ARCH_NAME}_${BLAS_mkl_OS_NAME}")

  foreach(IT ${BLAS_SEARCH_LIBS})
    string(REPLACE " " ";" SEARCH_LIBS ${IT})
    if(NOT ${_LIBRARIES})
      check_blas_libraries(
        ${_LIBRARIES}
        BLAS
        ${BLAS_mkl_SEARCH_SYMBOL}
        ""
        "${SEARCH_LIBS}"
        "${CMAKE_THREAD_LIBS_INIT};${BLAS_mkl_LM};${BLAS_mkl_LDL}"
        "${BLAS_mkl_MKLROOT}"
        "${BLAS_mkl_LIB_PATH_SUFFIXES}"
        )
    endif()
  endforeach()

  unset(BLAS_mkl_ILP_MODE)
  unset(BLAS_mkl_INTFACE)
  unset(BLAS_mkl_THREADING)
  unset(BLAS_mkl_OMP)
  unset(BLAS_mkl_DLL_SUFFIX)
  unset(BLAS_mkl_LM)
  unset(BLAS_mkl_LDL)
  unset(BLAS_mkl_MKLROOT)
  unset(BLAS_mkl_MKLROOT_LAST_DIR)
  unset(BLAS_mkl_ARCH_NAME)
  unset(BLAS_mkl_OS_NAME)
  unset(BLAS_mkl_LIB_PATH_SUFFIXES)
endif()


find_package_handle_standard_args(MKL REQUIRED_VARS MKL_LIBRARIES)

cmake_pop_check_state()
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_blas_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
