# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(faiss_c_library LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

set(FAISS_C_SRC
  AutoTune_c.cpp
  Clustering_c.cpp
  IndexFlat_c.cpp
  IndexIVFFlat_c.cpp
  IndexIVF_c.cpp
  IndexLSH_c.cpp
  IndexPreTransform_c.cpp
  VectorTransform_c.cpp
  IndexShards_c.cpp
  IndexReplicas_c.cpp
  Index_c.cpp
  IndexBinary_c.cpp
  IndexScalarQuantizer_c.cpp
  MetaIndexes_c.cpp
  clone_index_c.cpp
  error_impl.cpp
  index_factory_c.cpp
  index_io_c.cpp
  impl/AuxIndexStructures_c.cpp
  impl/io_c.cpp
  utils/distances_c.cpp
  utils/utils_c.cpp
)

add_library(faiss_c ${FAISS_C_SRC})
target_link_libraries(faiss_c PRIVATE faiss)

add_library(faiss_c_avx2 ${FAISS_C_SRC})
target_link_libraries(faiss_c_avx2 PRIVATE faiss_avx2)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(faiss_c_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(faiss_c_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
else()
  # MSVC enables FMA with /arch:AVX2; no separate flags for F16C, POPCNT
  # Ref. FMA (under /arch:AVX2): https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
  # Ref. F16C (2nd paragraph): https://walbourn.github.io/directxmath-avx2/
  # Ref. POPCNT: https://docs.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64
  target_compile_options(faiss_c_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

add_library(faiss_c_avx512 ${FAISS_C_SRC})
target_link_libraries(faiss_c_avx512 PRIVATE faiss_avx512)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(faiss_c_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  # All modern CPUs support F, CD, VL, DQ, BW extensions.
  # Ref: https://en.wikipedia.org/wiki/AVX512
  target_compile_options(faiss_c_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
else()
  target_compile_options(faiss_c_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
endif()

add_library(faiss_c_avx512_spr ${FAISS_C_SRC})
target_link_libraries(faiss_c_avx512_spr PRIVATE faiss_avx512_spr)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(faiss_c_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  # Architecture mode to support AVX512 extensions available since Intel(R) Sapphire Rapids.
  # Ref: https://networkbuilders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide
  target_compile_options(faiss_c_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
else()
  target_compile_options(faiss_c_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
endif()

add_library(faiss_c_sve ${FAISS_C_SRC})
target_link_libraries(faiss_c_sve PRIVATE faiss_sve)
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(faiss_c_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
    # Do nothing, expect SVE to be enabled by -march=native
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    # Add +sve
    target_compile_options(faiss_c_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
    # No valid -march, so specify -march=armv8-a+sve as the default
    target_compile_options(faiss_c_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
  endif()
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
    # Do nothing, expect SVE to be enabled by -march=native
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    # Add +sve
    target_compile_options(faiss_c_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
    # No valid -march, so specify -march=armv8-a+sve as the default
    target_compile_options(faiss_c_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
  endif()
endif()

function(faiss_install_headers headers p)
  foreach(h ${headers})
    get_filename_component(f ${h} DIRECTORY)
    install(FILES ${h}
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/${p}/${f}
    )
  endforeach()
endfunction()

file(GLOB FAISS_C_API_HEADERS
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     "*.h"
     "impl/*.h"
     "utils/*.h")

faiss_install_headers("${FAISS_C_API_HEADERS}" c_api)

install(TARGETS faiss_c
  EXPORT faiss-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
if(FAISS_OPT_LEVEL STREQUAL "avx2")
  install(TARGETS faiss_c_avx2
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "avx512")
  install(TARGETS faiss_c_avx2 faiss_c_avx512
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  install(TARGETS faiss_c_avx2 faiss_c_avx512_spr
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "sve")
  install(TARGETS faiss_c_sve
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

add_executable(example_c EXCLUDE_FROM_ALL example_c.c)
target_link_libraries(example_c PRIVATE faiss_c)

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    add_subdirectory(gpu-rocm)
  else ()
    add_subdirectory(gpu)
  endif()
endif()
