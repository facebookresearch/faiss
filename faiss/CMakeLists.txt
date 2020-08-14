# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(faiss
  AutoTune.cpp
  Clustering.cpp
  DirectMap.cpp
  IVFlib.cpp
  Index.cpp
  Index2Layer.cpp
  IndexBinary.cpp
  IndexBinaryFlat.cpp
  IndexBinaryFromFloat.cpp
  IndexBinaryHNSW.cpp
  IndexBinaryHash.cpp
  IndexBinaryIVF.cpp
  IndexFlat.cpp
  IndexHNSW.cpp
  IndexIVF.cpp
  IndexIVFFlat.cpp
  IndexIVFPQ.cpp
  IndexIVFPQR.cpp
  IndexIVFSpectralHash.cpp
  IndexLSH.cpp
  IndexLattice.cpp
  IndexPQ.cpp
  IndexPreTransform.cpp
  IndexReplicas.cpp
  IndexScalarQuantizer.cpp
  IndexShards.cpp
  InvertedLists.cpp
  MatrixStats.cpp
  MetaIndexes.cpp
  OnDiskInvertedLists.cpp
  VectorTransform.cpp
  clone_index.cpp
  index_factory.cpp
  impl/AuxIndexStructures.cpp
  impl/FaissException.cpp
  impl/HNSW.cpp
  impl/PolysemousTraining.cpp
  impl/ProductQuantizer.cpp
  impl/ScalarQuantizer.cpp
  impl/index_read.cpp
  impl/index_write.cpp
  impl/io.cpp
  impl/lattice_Zn.cpp
  utils/Heap.cpp
  utils/WorkerThread.cpp
  utils/distances.cpp
  utils/distances_simd.cpp
  utils/extra_distances.cpp
  utils/hamming.cpp
  utils/random.cpp
  utils/utils.cpp
)

set(FAISS_HEADERS
  AutoTune.h
  Clustering.h
  DirectMap.h
  IVFlib.h
  Index.h
  Index2Layer.h
  IndexBinary.h
  IndexBinaryFlat.h
  IndexBinaryFromFloat.h
  IndexBinaryHNSW.h
  IndexBinaryHash.h
  IndexBinaryIVF.h
  IndexFlat.h
  IndexHNSW.h
  IndexIVF.h
  IndexIVFFlat.h
  IndexIVFPQ.h
  IndexIVFPQR.h
  IndexIVFSpectralHash.h
  IndexLSH.h
  IndexLattice.h
  IndexPQ.h
  IndexPreTransform.h
  IndexReplicas.h
  IndexScalarQuantizer.h
  IndexShards.h
  InvertedLists.h
  MatrixStats.h
  MetaIndexes.h
  MetricType.h
  OnDiskInvertedLists.h
  VectorTransform.h
  clone_index.h
  index_factory.h
  index_io.h
  impl/AuxIndexStructures.h
  impl/FaissAssert.h
  impl/FaissException.h
  impl/HNSW.h
  impl/PolysemousTraining.h
  impl/ProductQuantizer-inl.h
  impl/ProductQuantizer.h
  impl/ScalarQuantizer.h
  impl/ThreadedIndex-inl.h
  impl/ThreadedIndex.h
  impl/io.h
  impl/io_macros.h
  impl/lattice_Zn.h
  utils/Heap.h
  utils/WorkerThread.h
  utils/distances.h
  utils/extra_distances.h
  utils/hamming-inl.h
  utils/hamming.h
  utils/random.h
  utils/utils.h
)

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mpopcnt>)
  set_target_properties(faiss PROPERTIES OUTPUT_NAME "faiss_avx2")
elseif(FAISS_OPT_LEVEL STREQUAL "sse4")
  target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse4 -mpopcnt>)
  set_target_properties(faiss PROPERTIES OUTPUT_NAME "faiss_sse4")
endif()

# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

# NOTE: This is needed for building faiss as a static library that will then be
# linked into a shared library.
set_property(TARGET faiss PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(faiss PRIVATE FINTEGER=int)

find_package(OpenMP REQUIRED)
target_link_libraries(faiss PRIVATE OpenMP::OpenMP_CXX)

find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(faiss PRIVATE ${MKL_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(faiss PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  target_link_libraries(faiss PRIVATE ${LAPACK_LIBRARIES})
endif()

install(TARGETS faiss
  EXPORT faiss-targets
  INCLUDES DESTINATION include
)
foreach(header ${FAISS_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION include/faiss/${dir}
  )
endforeach()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/faiss-config-version.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/faiss-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/faiss-config.cmake
  COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/faiss-config.cmake
  ${PROJECT_BINARY_DIR}/cmake/faiss-config-version.cmake
  DESTINATION share/faiss
)

install(EXPORT faiss-targets
  DESTINATION share/faiss
)
