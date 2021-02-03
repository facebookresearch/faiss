# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(faiss
  AutoTune.cpp
  Clustering.cpp
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
  IndexIVFPQFastScan.cpp
  IndexIVFPQR.cpp
  IndexIVFSpectralHash.cpp
  IndexLSH.cpp
  IndexLattice.cpp
  IndexPQ.cpp
  IndexPQFastScan.cpp
  IndexPreTransform.cpp
  IndexRefine.cpp
  IndexReplicas.cpp
  IndexScalarQuantizer.cpp
  IndexShards.cpp
  MatrixStats.cpp
  MetaIndexes.cpp
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
  impl/pq4_fast_scan.cpp
  impl/pq4_fast_scan_search_1.cpp
  impl/pq4_fast_scan_search_qbs.cpp
  invlists/BlockInvertedLists.cpp
  invlists/DirectMap.cpp
  invlists/InvertedLists.cpp
  invlists/InvertedListsIOHook.cpp
  utils/Heap.cpp
  utils/WorkerThread.cpp
  utils/distances.cpp
  utils/distances_simd.cpp
  utils/extra_distances.cpp
  utils/hamming.cpp
  utils/partitioning.cpp
  utils/quantize_lut.cpp
  utils/random.cpp
  utils/utils.cpp
)

set(FAISS_HEADERS
  AutoTune.h
  Clustering.h
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
  IndexIVFPQFastScan.h
  IndexIVFPQR.h
  IndexIVFSpectralHash.h
  IndexLSH.h
  IndexLattice.h
  IndexPQ.h
  IndexPQFastScan.h
  IndexPreTransform.h
  IndexRefine.h
  IndexReplicas.h
  IndexScalarQuantizer.h
  IndexShards.h
  MatrixStats.h
  MetaIndexes.h
  MetricType.h
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
  impl/ResultHandler.h
  impl/ScalarQuantizer.h
  impl/ThreadedIndex-inl.h
  impl/ThreadedIndex.h
  impl/io.h
  impl/io_macros.h
  impl/lattice_Zn.h
  impl/platform_macros.h
  impl/pq4_fast_scan.h
  impl/simd_result_handlers.h
  invlists/BlockInvertedLists.h
  invlists/DirectMap.h
  invlists/InvertedLists.h
  invlists/InvertedListsIOHook.h
  utils/AlignedTable.h
  utils/Heap.h
  utils/WorkerThread.h
  utils/distances.h
  utils/extra_distances.h
  utils/hamming-inl.h
  utils/hamming.h
  utils/ordered_key_value.h
  utils/partitioning.h
  utils/quantize_lut.h
  utils/random.h
  utils/simdlib.h
  utils/simdlib_avx2.h
  utils/simdlib_emulated.h
  utils/utils.h
)

# Export FAISS_HEADERS variable to parent scope.
set(FAISS_HEADERS ${FAISS_HEADERS} PARENT_SCOPE)

if(NOT WIN32)
  target_sources(faiss PRIVATE invlists/OnDiskInvertedLists.cpp)
  list(APPEND FAISS_HEADERS invlists/OnDiskInvertedLists.h)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
  set_target_properties(faiss PROPERTIES OUTPUT_NAME "faiss_avx2")
elseif(FAISS_OPT_LEVEL STREQUAL "sse4")
  target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse4 -mpopcnt>)
  set_target_properties(faiss PROPERTIES OUTPUT_NAME "faiss_sse4")
endif()

# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(faiss PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(faiss PRIVATE FAISS_MAIN_LIB)
endif()

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
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
foreach(header ${FAISS_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/${dir}
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
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/faiss
)

install(EXPORT faiss-targets
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/faiss
)
