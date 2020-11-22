# Install script for directory: /home/lvtingxun/Project/tools/faiss/faiss/gpu

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuAutoTune.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuCloner.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuClonerOptions.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuDistance.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuFaissAssert.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndexBinaryFlat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndexFlat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndex.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndexIVFFlat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndexIVF.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndexIVFPQ.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndexIVFScalarQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuIndicesOptions.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/GpuResources.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/StandardGpuResources.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/BinaryDistance.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/BinaryFlatIndex.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/BroadcastSum.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/Distance.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/DistanceUtils.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/FlatIndex.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/GeneralDistance.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/GpuScalarQuantizer.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/IVFAppend.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/IVFBase.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/IVFFlat.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/IVFFlatScan.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/IVFPQ.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/IVFUtils.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/L2Norm.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/L2Select.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/PQCodeDistances.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/PQCodeDistances-inl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/PQCodeLoad.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/RemapIndices.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/impl" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/impl/VectorResidual.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils/blockselect" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/blockselect/BlockSelectImpl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/BlockSelectKernel.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Comparators.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/ConversionOperators.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/CopyUtils.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/DeviceDefs.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/DeviceTensor.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/DeviceTensor-inl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/DeviceUtils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/DeviceVector.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Float16.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/HostTensor.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/HostTensor-inl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Limits.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/LoadStoreOperators.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/MathOperators.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/MatrixMult.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/MatrixMult-inl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/MergeNetworkBlock.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/MergeNetworkUtils.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/MergeNetworkWarp.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/NoTypeTensor.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Pair.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/PtxUtils.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/ReductionOperators.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Reductions.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Select.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/StackDeviceMemory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/StaticUtils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Tensor.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Tensor-inl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/ThrustAllocator.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Timer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/Transpose.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/WarpSelectKernel.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils/warpselect" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/warpselect/WarpSelectImpl.cuh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/gpu/utils" TYPE FILE FILES "/home/lvtingxun/Project/tools/faiss/faiss/gpu/utils/WarpShuffles.cuh")
endif()

