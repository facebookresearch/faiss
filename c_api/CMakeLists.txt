# Copyright (c) Facebook, Inc. and its affiliates.
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
  Index_c.cpp
  IndexScalarQuantizer_c.cpp
  MetaIndexes_c.cpp
  clone_index_c.cpp
  error_impl.cpp
  index_factory_c.cpp
  index_io_c.cpp
  impl/AuxIndexStructures_c.cpp
  utils/distances_c.cpp
)
add_library(faiss_c ${FAISS_C_SRC})
target_link_libraries(faiss_c PRIVATE faiss)

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

add_library(faiss_avx2_c ${FAISS_C_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(faiss_avx2_c PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
target_link_libraries(faiss_avx2_c PRIVATE faiss_avx2)

add_executable(example_c EXCLUDE_FROM_ALL example_c.c)
if(FAISS_OPT_LEVEL STREQUAL "avx2")
  target_link_libraries(example_c PRIVATE faiss_avx2_c)
else()
  target_link_libraries(example_c PRIVATE faiss_c)
endif()

if(FAISS_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
