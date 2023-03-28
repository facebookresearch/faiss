# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_TEST_SRC
  test_binary_flat.cpp
  test_dealloc_invlists.cpp
  test_ivfpq_codec.cpp
  test_ivfpq_indexing.cpp
  test_lowlevel_ivf.cpp
  test_omp_threads.cpp
  test_pairs_decoding.cpp
  test_params_override.cpp
  test_pq_encoding.cpp
  test_sliding_ivf.cpp
  test_threaded_index.cpp
  test_transfer_invlists.cpp
  test_mem_leak.cpp
  test_cppcontrib_sa_decode.cpp
  test_cppcontrib_uintreader.cpp
  test_simdlib.cpp
  test_approx_topk.cpp
  test_RCQ_cropping.cpp
)
if ( NOT WIN32 ) 
  set(FAISS_TEST_SRC
    ${FAISS_TEST_SRC}
    test_merge.cpp
    test_ondisk_ivf.cpp
  )
endif ()

add_executable(faiss_test ${FAISS_TEST_SRC})

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  if(NOT WIN32)
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
  else()
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  endif()
  target_link_libraries(faiss_test PRIVATE faiss_avx2)
else()
  target_link_libraries(faiss_test PRIVATE faiss)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.12.1.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
if ( MSVC ) 
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

target_link_libraries(faiss_test PRIVATE
  OpenMP::OpenMP_CXX
  gtest_main
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_test)

function(copyFrameworkFiles CC_FRAMEWORK_RUNTIME_FILES DEP)
    foreach(FRAMEWORK_FILE ${CC_FRAMEWORK_RUNTIME_FILES})
        get_filename_component(OUTFILENAME ${FRAMEWORK_FILE} NAME)
        if ( MSVC )
            set(OUTPATH "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${OUTFILENAME}")
        else()
            set(OUTPATH "${CMAKE_CURRENT_BINARY_DIR}/${OUTFILENAME}")
        endif()
        # Make sure the CC_FRAMEWORK_RUNTIME_FILES files have a newer timestamp than 
        # the zip they came from so that cmake doesn't re-copy them over and over
        add_custom_command(
            OUTPUT "${FRAMEWORK_FILE}"
            COMMAND ${CMAKE_COMMAND} -E touch_nocreate "${FRAMEWORK_FILE}"
            DEPENDS "${DEP}"
            COMMENT "Touching ${FRAMEWORK_FILE}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${OUTPATH}"
            COMMAND ${CMAKE_COMMAND} -E copy "${FRAMEWORK_FILE}" "${OUTPATH}"
            DEPENDS "${FRAMEWORK_FILE}" "${DEP}"
            VERBATIM
        )
        list(APPEND COPYFRAMEWORKFILES "${OUTPATH}")
    endforeach()
    add_custom_target(CopyFrameworkFiles DEPENDS ${COPYFRAMEWORKFILES})
    set_property(TARGET CopyFrameworkFiles PROPERTY FOLDER ZZ_Dependencies)
endfunction()
# no idea how to find the test dlls. 
# copyFrameworkFiles("$<TARGET_FILE:googletest/gtest_main>" gtest)
