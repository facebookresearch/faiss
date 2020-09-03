# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

project(faiss_c_library LANGUAGES C CXX)

add_library(faiss_c)

target_sources(faiss_c PRIVATE
  AutoTune_c.cpp
  clone_index_c.cpp
  Clustering_c.cpp
  error_impl.cpp
  Index_c.cpp
  index_factory_c.cpp
  IndexFlat_c.cpp
  index_io_c.cpp
  IndexIVF_c.cpp
  IndexIVFFlat_c.cpp
  IndexLSH_c.cpp
  IndexPreTransform_c.cpp
  IndexShards_c.cpp
  MetaIndexes_c.cpp
)

set(FAISS_C_API_HEADERS
  AutoTune_c.h
  clone_index_c.h
  Clustering_c.h
  error_c.h
  error_impl.h
  faiss_c.h
  Index_c.h
  index_factory_c.h
  IndexFlat_c.h
  index_io_c.h
  IndexIVF_c.h
  IndexIVFFlat_c.h
  IndexLSH_c.h
  IndexPreTransform_c.h
  IndexShards_c.h
  macros_impl.h
  MetaIndexes_c.h
)

add_dependencies(faiss_c faiss)
target_include_directories(faiss_c PRIVATE ${CMAKE_SOURCE_DIR})

foreach(header ${FAISS_C_API_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION include/faiss/c_api/${dir}
  )
endforeach()

add_executable(example_c)
target_sources(example_c PRIVATE example_c.c)
set_target_properties(example_c PROPERTIES LANGUAGE C)
target_link_libraries(example_c PRIVATE faiss_c)
target_link_libraries(example_c PRIVATE faiss)
