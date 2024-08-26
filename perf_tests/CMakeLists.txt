project(faiss_perf_tests)


include(FetchContent)
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark

set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(
  googlebenchmark)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB files "${PROJECT_SOURCE_DIR}/*.cpp")
include(CMakePrintHelpers)

cmake_print_variables(files)

cmake_print_variables(PROJECT_SOURCE_DIR)

foreach(ONE_BENCH_CPP ${files})
  get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)
  set(TARGET_NAME bench_${ONE_BENCH_EXEC})
  cmake_print_variables(ONE_BENCH_EXEC)

  add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})

  if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "sve")
    target_link_libraries(${TARGET_NAME} PRIVATE faiss)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx2")
    if(NOT WIN32)
      target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    else()
      target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE faiss_avx2)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "avx512")
    if(NOT WIN32)
      target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
    else()
      target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE faiss_avx512)
  endif()

  if(FAISS_OPT_LEVEL STREQUAL "sve")
    if(NOT WIN32)
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
      endif()
      if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
        # Do nothing, expect SVE to be enabled by -march=native
      elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
        # Add +sve
        target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
      elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
        # No valid -march, so specify -march=armv8-a+sve as the default
        target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
      endif()
    else()
      # TODO: support Windows
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE faiss_sve)
  endif()


target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX benchmark::benchmark)
endforeach()
