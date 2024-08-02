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
  utils/distances_c.cpp
  utils/utils_c.cpp
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

add_executable(example_c EXCLUDE_FROM_ALL example_c.c)
target_link_libraries(example_c PRIVATE faiss_c)

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    add_subdirectory(gpu-rocm)
  else ()
    add_subdirectory(gpu)
  endif()
endif()
