# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_GPU_SRC
  GpuAutoTune.cpp
  GpuCloner.cpp
  GpuClonerOptions.cpp
  GpuDistance.cu
  GpuIcmEncoder.cu
  GpuIndex.cu
  GpuIndexBinaryFlat.cu
  GpuIndexFlat.cu
  GpuIndexIVF.cu
  GpuIndexIVFFlat.cu
  GpuIndexIVFPQ.cu
  GpuIndexIVFScalarQuantizer.cu
  GpuResources.cpp
  StandardGpuResources.cpp
  impl/BinaryDistance.cu
  impl/BinaryFlatIndex.cu
  impl/BroadcastSum.cu
  impl/Distance.cu
  impl/FlatIndex.cu
  impl/IndexUtils.cu
  impl/IVFAppend.cu
  impl/IVFBase.cu
  impl/IVFFlat.cu
  impl/IVFFlatScan.cu
  impl/IVFInterleaved.cu
  impl/IVFPQ.cu
  impl/IVFUtils.cu
  impl/IVFUtilsSelect1.cu
  impl/IVFUtilsSelect2.cu
  impl/InterleavedCodes.cpp
  impl/L2Norm.cu
  impl/L2Select.cu
  impl/PQScanMultiPassPrecomputed.cu
  impl/RemapIndices.cpp
  impl/VectorResidual.cu
  impl/scan/IVFInterleaved1.cu
  impl/scan/IVFInterleaved32.cu
  impl/scan/IVFInterleaved64.cu
  impl/scan/IVFInterleaved128.cu
  impl/scan/IVFInterleaved256.cu
  impl/scan/IVFInterleaved512.cu
  impl/scan/IVFInterleaved1024.cu
  impl/scan/IVFInterleaved2048.cu
  impl/IcmEncoder.cu
  utils/BlockSelectFloat.cu
  utils/DeviceUtils.cu
  utils/StackDeviceMemory.cpp
  utils/Timer.cpp
  utils/WarpSelectFloat.cu
  utils/blockselect/BlockSelectFloat1.cu
  utils/blockselect/BlockSelectFloat32.cu
  utils/blockselect/BlockSelectFloat64.cu
  utils/blockselect/BlockSelectFloat128.cu
  utils/blockselect/BlockSelectFloat256.cu
  utils/blockselect/BlockSelectFloatF512.cu
  utils/blockselect/BlockSelectFloatF1024.cu
  utils/blockselect/BlockSelectFloatF2048.cu
  utils/blockselect/BlockSelectFloatT512.cu
  utils/blockselect/BlockSelectFloatT1024.cu
  utils/blockselect/BlockSelectFloatT2048.cu
  utils/warpselect/WarpSelectFloat1.cu
  utils/warpselect/WarpSelectFloat32.cu
  utils/warpselect/WarpSelectFloat64.cu
  utils/warpselect/WarpSelectFloat128.cu
  utils/warpselect/WarpSelectFloat256.cu
  utils/warpselect/WarpSelectFloatF512.cu
  utils/warpselect/WarpSelectFloatF1024.cu
  utils/warpselect/WarpSelectFloatF2048.cu
  utils/warpselect/WarpSelectFloatT512.cu
  utils/warpselect/WarpSelectFloatT1024.cu
  utils/warpselect/WarpSelectFloatT2048.cu
)

set(FAISS_GPU_HEADERS
  GpuAutoTune.h
  GpuCloner.h
  GpuClonerOptions.h
  GpuDistance.h
  GpuIcmEncoder.h
  GpuFaissAssert.h
  GpuIndex.h
  GpuIndexBinaryFlat.h
  GpuIndexFlat.h
  GpuIndexIVF.h
  GpuIndexIVFFlat.h
  GpuIndexIVFPQ.h
  GpuIndexIVFScalarQuantizer.h
  GpuIndicesOptions.h
  GpuResources.h
  StandardGpuResources.h
  impl/BinaryDistance.cuh
  impl/BinaryFlatIndex.cuh
  impl/BroadcastSum.cuh
  impl/Distance.cuh
  impl/DistanceUtils.cuh
  impl/FlatIndex.cuh
  impl/GeneralDistance.cuh
  impl/GpuScalarQuantizer.cuh
  impl/IndexUtils.h
  impl/IVFAppend.cuh
  impl/IVFBase.cuh
  impl/IVFFlat.cuh
  impl/IVFFlatScan.cuh
  impl/IVFInterleaved.cuh
  impl/IVFPQ.cuh
  impl/IVFUtils.cuh
  impl/InterleavedCodes.h
  impl/L2Norm.cuh
  impl/L2Select.cuh
  impl/PQCodeDistances-inl.cuh
  impl/PQCodeDistances.cuh
  impl/PQCodeLoad.cuh
  impl/PQScanMultiPassNoPrecomputed-inl.cuh
  impl/PQScanMultiPassNoPrecomputed.cuh
  impl/PQScanMultiPassPrecomputed.cuh
  impl/RemapIndices.h
  impl/VectorResidual.cuh
  impl/scan/IVFInterleavedImpl.cuh
  impl/IcmEncoder.cuh
  utils/BlockSelectKernel.cuh
  utils/Comparators.cuh
  utils/ConversionOperators.cuh
  utils/CopyUtils.cuh
  utils/DeviceDefs.cuh
  utils/DeviceTensor-inl.cuh
  utils/DeviceTensor.cuh
  utils/DeviceUtils.h
  utils/DeviceVector.cuh
  utils/Float16.cuh
  utils/HostTensor-inl.cuh
  utils/HostTensor.cuh
  utils/Limits.cuh
  utils/LoadStoreOperators.cuh
  utils/MathOperators.cuh
  utils/MatrixMult-inl.cuh
  utils/MatrixMult.cuh
  utils/MergeNetworkBlock.cuh
  utils/MergeNetworkUtils.cuh
  utils/MergeNetworkWarp.cuh
  utils/NoTypeTensor.cuh
  utils/Pair.cuh
  utils/PtxUtils.cuh
  utils/ReductionOperators.cuh
  utils/Reductions.cuh
  utils/Select.cuh
  utils/StackDeviceMemory.h
  utils/StaticUtils.h
  utils/Tensor-inl.cuh
  utils/Tensor.cuh
  utils/ThrustUtils.cuh
  utils/Timer.h
  utils/Transpose.cuh
  utils/WarpPackedBits.cuh
  utils/WarpSelectKernel.cuh
  utils/WarpShuffles.cuh
  utils/blockselect/BlockSelectImpl.cuh
  utils/warpselect/WarpSelectImpl.cuh
)

# Export FAISS_GPU_HEADERS variable to parent scope.
set(FAISS_GPU_HEADERS ${FAISS_GPU_HEADERS} PARENT_SCOPE)

target_sources(faiss PRIVATE ${FAISS_GPU_SRC})
target_sources(faiss_avx2 PRIVATE ${FAISS_GPU_SRC})

foreach(header ${FAISS_GPU_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/${dir}
  )
endforeach()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(faiss PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(faiss_avx2 PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)
target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)
