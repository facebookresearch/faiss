# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_SRC
  AutoTune.cpp
  Clustering.cpp
  IVFlib.cpp
  Index.cpp
  Index2Layer.cpp
  IndexAdditiveQuantizer.cpp
  IndexBinary.cpp
  IndexBinaryFlat.cpp
  IndexBinaryFromFloat.cpp
  IndexBinaryHNSW.cpp
  IndexBinaryHash.cpp
  IndexBinaryIVF.cpp
  IndexFlat.cpp
  IndexFlatCodes.cpp
  IndexHNSW.cpp
  IndexIDMap.cpp
  IndexIVF.cpp
  IndexIVFAdditiveQuantizer.cpp
  IndexIVFFlat.cpp
  IndexIVFPQ.cpp
  IndexIVFFastScan.cpp
  IndexIVFAdditiveQuantizerFastScan.cpp
  IndexIVFPQFastScan.cpp
  IndexIVFPQR.cpp
  IndexIVFSpectralHash.cpp
  IndexLSH.cpp
  IndexNNDescent.cpp
  IndexLattice.cpp
  IndexNSG.cpp
  IndexPQ.cpp
  IndexFastScan.cpp
  IndexAdditiveQuantizerFastScan.cpp
  IndexIVFIndependentQuantizer.cpp
  IndexPQFastScan.cpp
  IndexPreTransform.cpp
  IndexRefine.cpp
  IndexReplicas.cpp
  IndexRowwiseMinMax.cpp
  IndexScalarQuantizer.cpp
  IndexShards.cpp
  IndexShardsIVF.cpp
  IndexNeuralNetCodec.cpp
  MatrixStats.cpp
  MetaIndexes.cpp
  VectorTransform.cpp
  clone_index.cpp
  index_factory.cpp
  impl/AuxIndexStructures.cpp
  impl/CodePacker.cpp
  impl/IDSelector.cpp
  impl/FaissException.cpp
  impl/HNSW.cpp
  impl/NSG.cpp
  impl/PolysemousTraining.cpp
  impl/ProductQuantizer.cpp
  impl/AdditiveQuantizer.cpp
  impl/ResidualQuantizer.cpp
  impl/LocalSearchQuantizer.cpp
  impl/ProductAdditiveQuantizer.cpp
  impl/ScalarQuantizer.cpp
  impl/index_read.cpp
  impl/index_write.cpp
  impl/io.cpp
  impl/kmeans1d.cpp
  impl/lattice_Zn.cpp
  impl/pq4_fast_scan.cpp
  impl/pq4_fast_scan_search_1.cpp
  impl/pq4_fast_scan_search_qbs.cpp
  impl/residual_quantizer_encode_steps.cpp
  impl/io.cpp
  impl/lattice_Zn.cpp
  impl/NNDescent.cpp
  invlists/BlockInvertedLists.cpp
  invlists/DirectMap.cpp
  invlists/InvertedLists.cpp
  invlists/InvertedListsIOHook.cpp
  utils/Heap.cpp
  utils/NeuralNet.cpp
  utils/WorkerThread.cpp
  utils/distances.cpp
  utils/distances_simd.cpp
  utils/extra_distances.cpp
  utils/hamming.cpp
  utils/partitioning.cpp
  utils/quantize_lut.cpp
  utils/random.cpp
  utils/sorting.cpp
  utils/utils.cpp
  utils/distances_fused/avx512.cpp
  utils/distances_fused/distances_fused.cpp
  utils/distances_fused/simdlib_based.cpp
)

set(FAISS_HEADERS
  AutoTune.h
  Clustering.h
  IVFlib.h
  Index.h
  Index2Layer.h
  IndexAdditiveQuantizer.h
  IndexBinary.h
  IndexBinaryFlat.h
  IndexBinaryFromFloat.h
  IndexBinaryHNSW.h
  IndexBinaryHash.h
  IndexBinaryIVF.h
  IndexFlat.h
  IndexFlatCodes.h
  IndexHNSW.h
  IndexIDMap.h
  IndexIVF.h
  IndexIVFAdditiveQuantizer.h
  IndexIVFIndependentQuantizer.h
  IndexIVFFlat.h
  IndexIVFPQ.h
  IndexIVFFastScan.h
  IndexIVFAdditiveQuantizerFastScan.h
  IndexIVFPQFastScan.h
  IndexIVFPQR.h
  IndexIVFSpectralHash.h
  IndexLSH.h
  IndexNeuralNetCodec.h
  IndexLattice.h
  IndexNNDescent.h
  IndexNSG.h
  IndexPQ.h
  IndexFastScan.h
  IndexAdditiveQuantizerFastScan.h
  IndexPQFastScan.h
  IndexPreTransform.h
  IndexRefine.h
  IndexReplicas.h
  IndexRowwiseMinMax.h
  IndexScalarQuantizer.h
  IndexShards.h
  IndexShardsIVF.h
  MatrixStats.h
  MetaIndexes.h
  MetricType.h
  VectorTransform.h
  clone_index.h
  index_factory.h
  index_io.h
  impl/AdditiveQuantizer.h
  impl/AuxIndexStructures.h
  impl/CodePacker.h
  impl/IDSelector.h
  impl/DistanceComputer.h
  impl/FaissAssert.h
  impl/FaissException.h
  impl/HNSW.h
  impl/LocalSearchQuantizer.h
  impl/ProductAdditiveQuantizer.h
  impl/LookupTableScaler.h
  impl/NNDescent.h
  impl/NSG.h
  impl/PolysemousTraining.h
  impl/ProductQuantizer-inl.h
  impl/ProductQuantizer.h
  impl/Quantizer.h
  impl/ResidualQuantizer.h
  impl/ResultHandler.h
  impl/ScalarQuantizer.h
  impl/ThreadedIndex-inl.h
  impl/ThreadedIndex.h
  impl/index_read_utils.h
  impl/io.h
  impl/io_macros.h
  impl/kmeans1d.h
  impl/lattice_Zn.h
  impl/platform_macros.h
  impl/pq4_fast_scan.h
  impl/residual_quantizer_encode_steps.h
  impl/simd_result_handlers.h
  impl/code_distance/code_distance.h
  impl/code_distance/code_distance-generic.h
  impl/code_distance/code_distance-avx2.h
  impl/code_distance/code_distance-avx512.h
  impl/code_distance/code_distance-sve.h
  invlists/BlockInvertedLists.h
  invlists/DirectMap.h
  invlists/InvertedLists.h
  invlists/InvertedListsIOHook.h
  invlists/OnDiskInvertedLists.h
  utils/AlignedTable.h
  utils/bf16.h
  utils/Heap.h
  utils/NeuralNet.h
  utils/WorkerThread.h
  utils/distances.h
  utils/extra_distances-inl.h
  utils/extra_distances.h
  utils/fp16-fp16c.h
  utils/fp16-inl.h
  utils/fp16-arm.h
  utils/fp16.h
  utils/hamming-inl.h
  utils/hamming.h
  utils/ordered_key_value.h
  utils/partitioning.h
  utils/prefetch.h
  utils/quantize_lut.h
  utils/random.h
  utils/sorting.h
  utils/simdlib.h
  utils/simdlib_avx2.h
  utils/simdlib_avx512.h
  utils/simdlib_emulated.h
  utils/simdlib_neon.h
  utils/simdlib_ppc64.h
  utils/utils.h
  utils/distances_fused/avx512.h
  utils/distances_fused/distances_fused.h
  utils/distances_fused/simdlib_based.h
  utils/approx_topk/approx_topk.h
  utils/approx_topk/avx2-inl.h
  utils/approx_topk/generic.h
  utils/approx_topk/mode.h
  utils/approx_topk_hamming/approx_topk_hamming.h
  utils/transpose/transpose-avx2-inl.h
  utils/transpose/transpose-avx512-inl.h
  utils/hamming_distance/common.h
  utils/hamming_distance/generic-inl.h
  utils/hamming_distance/hamdis-inl.h
  utils/hamming_distance/neon-inl.h
  utils/hamming_distance/avx2-inl.h
  utils/hamming_distance/avx512-inl.h
)

if(NOT WIN32)
  list(APPEND FAISS_SRC invlists/OnDiskInvertedLists.cpp)
  list(APPEND FAISS_HEADERS invlists/OnDiskInvertedLists.h)
endif()

# Export FAISS_HEADERS variable to parent scope.
set(FAISS_HEADERS ${FAISS_HEADERS} PARENT_SCOPE)

add_library(faiss ${FAISS_SRC})

add_library(faiss_avx2 ${FAISS_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
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
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

add_library(faiss_avx512 ${FAISS_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(faiss_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  # All modern CPUs support F, CD, VL, DQ, BW extensions.
  # Ref: https://en.wikipedia.org/wiki/AVX512
  target_compile_options(faiss_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
else()
  target_compile_options(faiss_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

add_library(faiss_avx512_spr ${FAISS_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(faiss_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  # Architecture mode to support AVX512 extensions available since Intel(R) Sapphire Rapids.
  # Ref: https://networkbuilders.intel.com/solutionslibrary/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide
  target_compile_options(faiss_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=sapphirerapids -mtune=sapphirerapids>)
else()
  target_compile_options(faiss_avx512_spr PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  # we need bigobj for the swig wrapper
  add_compile_options(/bigobj)
endif()

add_library(faiss_sve ${FAISS_SRC})
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(faiss_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
if(NOT WIN32)
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
    # Do nothing, expect SVE to be enabled by -march=native
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    # Add +sve
    target_compile_options(faiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
    # No valid -march, so specify -march=armv8-a+sve as the default
    target_compile_options(faiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
  endif()
  if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
    # Do nothing, expect SVE to be enabled by -march=native
  elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
    # Add +sve
    target_compile_options(faiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
  elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
    # No valid -march, so specify -march=armv8-a+sve as the default
    target_compile_options(faiss_sve PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
  endif()
endif()

# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss_avx2 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss_avx512 PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss_avx512_spr PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <faiss/foo.h>`.
target_include_directories(faiss_sve PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(faiss faiss_avx2 faiss_avx512 faiss_avx512_spr faiss_sve PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if(WIN32)
  target_compile_definitions(faiss PRIVATE FAISS_MAIN_LIB)
  target_compile_definitions(faiss_avx2 PRIVATE FAISS_MAIN_LIB)
  target_compile_definitions(faiss_avx512 PRIVATE FAISS_MAIN_LIB)
  target_compile_definitions(faiss_avx512_spr PRIVATE FAISS_MAIN_LIB)
  target_compile_definitions(faiss_sve PRIVATE FAISS_MAIN_LIB)
endif()

string(FIND "${CMAKE_CXX_FLAGS}" "FINTEGER" finteger_idx)
if (${finteger_idx} EQUAL -1)
  target_compile_definitions(faiss PRIVATE FINTEGER=int)
endif()
target_compile_definitions(faiss_avx2 PRIVATE FINTEGER=int)
target_compile_definitions(faiss_avx512 PRIVATE FINTEGER=int)
target_compile_definitions(faiss_avx512_spr PRIVATE FINTEGER=int)
target_compile_definitions(faiss_sve PRIVATE FINTEGER=int)

if(FAISS_USE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

  if (ipo_supported)
    message(STATUS "LTO enabled")
    set_property(TARGET faiss PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET faiss_avx2 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET faiss_avx512 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET faiss_avx512_spr PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "LTO not supported: <${ipo_error}>")
  endif()
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(faiss PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_avx2 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_avx512 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_avx512_spr PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_sve PRIVATE OpenMP::OpenMP_CXX)

if(FAISS_ENABLE_MKL)
  find_package(MKL)
endif()
if(MKL_FOUND)
  target_link_libraries(faiss PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(faiss_avx2 PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(faiss_avx512 PRIVATE ${MKL_LIBRARIES})
  target_link_libraries(faiss_avx512_spr PRIVATE ${MKL_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(faiss PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(faiss_avx2 PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(faiss_avx512 PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(faiss_avx512_spr PRIVATE ${BLAS_LIBRARIES})
  target_link_libraries(faiss_sve PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  target_link_libraries(faiss PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(faiss_avx2 PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(faiss_avx512 PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(faiss_avx512_spr PRIVATE ${LAPACK_LIBRARIES})
  target_link_libraries(faiss_sve PRIVATE ${LAPACK_LIBRARIES})
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
if(FAISS_OPT_LEVEL STREQUAL "avx512")
  install(TARGETS faiss_avx2 faiss_avx512
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  install(TARGETS faiss_avx2 faiss_avx512 faiss_avx512_spr
    EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
if(FAISS_OPT_LEVEL STREQUAL "sve")
  install(TARGETS faiss_sve
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
