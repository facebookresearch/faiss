# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

function(link_to_faiss_lib target)
  if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr" AND NOT FAISS_OPT_LEVEL STREQUAL "sve" AND NOT FAISS_OPT_LEVEL STREQUAL "dd")
    target_link_libraries(${target} PRIVATE faiss)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx2")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    target_link_libraries(${target} PRIVATE faiss_avx2)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx512")
    if(NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE faiss_avx512)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
    if(NOT WIN32)
      # Architecture mode to support AVX512 extensions available since Intel (R) Sapphire Rapids.
      # Ref: https://networkbuilders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
    else()
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${target} PRIVATE faiss_avx512_spr)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "sve")
    if(NOT WIN32)
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
      endif()
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${target}  PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
      endif()
    else()
      # TODO: support Windows
    endif()
    target_link_libraries(${target} PRIVATE faiss_sve)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "dd")
    # DD mode: link to main faiss library with DD-specific definitions
    # When FAISS_OPT_LEVEL=dd, the main faiss library is built with DD enabled,
    # so we link to faiss (not a separate faiss_dd).
    # FAISS_ENABLE_DD exposes SIMDConfig class to consuming code (e.g., tests)
    # COMPILE_SIMD_* flags enable DD code paths in headers (architecture-specific)
    # Note: No SIMD compile flags here - DD handles dispatch internally.
    # Special tests (like test_simd_levels.cpp) that use raw intrinsics
    # should get their own SIMD flags via set_source_files_properties.
    #
    # Architecture-specific definitions mirror simd_dispatch.bzl dispatch_config:
    # - x86_64: AVX2 + AVX512 enabled
    # - aarch64: ARM_NEON enabled
    target_compile_definitions(${target} PRIVATE FAISS_ENABLE_DD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64)")
      target_compile_definitions(${target} PRIVATE COMPILE_SIMD_AVX2 COMPILE_SIMD_AVX512)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
      target_compile_definitions(${target} PRIVATE COMPILE_SIMD_ARM_NEON)
    endif()
    target_link_libraries(${target} PRIVATE faiss)
  endif()
endfunction()
