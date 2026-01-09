cmake_minimum_required(VERSION 3.24.0)

project(faiss_cppcontrib_knowhere)

find_package(OpenMP REQUIRED)

set(SRC_FAISS_CPPCONTRIB_KNOWHERE
    IndexBruteForceWrapper.cpp
    IndexHNSWWrapper.cpp
    IndexWrapper.cpp
)

include(${PROJECT_SOURCE_DIR}/../../../cmake/link_to_faiss_lib.cmake)

add_library(faiss_cppcontrib_knowhere ${SRC_FAISS_CPPCONTRIB_KNOWHERE})
link_to_faiss_lib(faiss_cppcontrib_knowhere)

target_include_directories(faiss_cppcontrib_knowhere PRIVATE 
    ${PROJECT_SOURCE_DIR}/../../..
)

add_executable(bench_hnsw_knowhere benchs/bench_hnsw_knowhere.cpp)

link_to_faiss_lib(bench_hnsw_knowhere)
target_link_libraries(bench_hnsw_knowhere PRIVATE faiss_cppcontrib_knowhere OpenMP::OpenMP_CXX)

target_include_directories(bench_hnsw_knowhere PRIVATE 
    ${PROJECT_SOURCE_DIR}/../../..
)
