project(faiss_perf_tests)


include(DownloadProject)
download_project(PROJ                googlebenchmark
                 GIT_REPOSITORY      https://github.com/google/benchmark.git
                 GIT_TAG             master
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)
add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
include_directories("${googlebenchmark_SOURCE_DIR}/include")

find_package(GoogleBenchmark REQUIRED)

file(GLOB_RECURSE ALL_BENCH_CPP *.cpp)

foreach(ONE_BENCH_CPP ${ALL_BENCH_CPP})

   get_filename_component(ONE_BENCH_EXEC ${ONE_BENCH_CPP} NAME_WE)

   # Avoid name collision 
   set(TARGET_NAME Bench_${ONE_BENCH_EXEC})

   add_executable(${TARGET_NAME} ${ONE_BENCH_CPP})
   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
   target_link_libraries(${TARGET_NAME} 
     ${CMAKE_THREAD_LIBS_INIT})

   # If you want to run benchmarks with the "make test" command, uncomment me
   add_test(${TARGET_NAME} ${ONE_BENCH_EXEC})
endforeach()
