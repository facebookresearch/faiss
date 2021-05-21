# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_SRC
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
  IndexNNDescent.cpp
  IndexLattice.cpp
  IndexNSG.cpp
  IndexPQ.cpp
  IndexPQFastScan.cpp
  IndexPreTransform.cpp
  IndexRefine.cpp
  IndexReplicas.cpp
  IndexResidual.cpp
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
  impl/NSG.cpp
  impl/PolysemousTraining.cpp
  impl/ProductQuantizer.cpp
  impl/AdditiveQuantizer.cpp
  impl/ResidualQuantizer.cpp
  impl/LocalSearchQuantizer.cpp
  impl/ScalarQuantizer.cpp
  impl/index_read.cpp
  impl/index_write.cpp
  impl/io.cpp
  impl/lattice_Zn.cpp
  impl/pq4_fast_scan.cpp
  impl/pq4_fast_scan_search_1.cpp
  impl/pq4_fast_scan_search_qbs.cpp
  impl/io.cpp
  impl/lattice_Zn.cpp
  impl/NNDescent.cpp
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
  IndexNNDescent.h
  IndexNSG.h
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
  impl/NSG.h
  impl/PolysemousTraining.h
  impl/ProductQuantizer-inl.h
  impl/ProductQuantizer.h
  impl/AdditiveQuantizer.h
  impl/ResidualQuantizer.h
  impl/LocalSearchQuantizer.h
  impl/ResultHandler.h
  impl/ScalarQuantizer.h
  impl/ThreadedIndex-inl.h
  impl/ThreadedIndex.h
  impl/io.h
  impl/io_macros.h
  impl/lattice_Zn.h
  impl/NNDescent.h
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
  utils/simdlib_neon.h
  utils/utils.h
)

if(NOT WIN32)
  list(APPEND FAISS_SRC invlists/OnDiskInvertedLists.cpp)
  list(APPEND FAISS_HEADERS invlists/OnDiskInvertedLists.h)
endif()

# Export FAISS_HEADERS variable to parent scope.
set(FAISS_HEADERS ${FAISS_HEADERS} PARENT_SCOPE)

add_library(faiss ${FAISS_SRC})

add_library(faiss_avx2 ${FAISS_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(faiss_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
else()
  # MSVC enables FMA with /arch:AVX2; no separate flags for F16C, POPCNT
  # Ref. FMA (under /arch:AVX2): https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
  # Ref. F16C (2nd paragraph): https://walbourn.github.io/directxmath-avx2/
  # Ref. POPCNT: https://docs.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64
  target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(faiss PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
set_target_properties(faiss_avx2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(faiss PRIVATE FAISS_MAIN_LIB)
  target_compile_definitions(faiss_avx2 PRIVATE FAISS_MAIN_LIB)
endif()

target_compile_definitions(faiss PRIVATE FINTEGER=int)
target_compile_definitions(faiss_avx2 PRIVATE FINTEGER=int)

find_package(OpenMP REQUIRED)
target_link_libraries(faiss PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_avx2 PRIVATE OpenMP::OpenMP_CXX)

find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(faiss PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(faiss_avx2 PRIVATE ${MKL_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(faiss PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(faiss_avx2 PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  target_link_libraries(faiss PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(faiss_avx2 PRIVATE ${LAPACK_LIBRARIES})
endif()

install(TARGETS faiss
  EXPORT faiss-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(FAISS_OPT_LEVEL STREQUAL "avx2")
  install(TARGETS faiss_avx2
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

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
