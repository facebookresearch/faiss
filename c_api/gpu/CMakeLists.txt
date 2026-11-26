# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_GPU_C_SRC
  DeviceUtils_c.cpp
  GpuAutoTune_c.cpp
  GpuClonerOptions_c.cpp
  GpuIndex_c.cpp
  GpuResources_c.cpp
  StandardGpuResources_c.cpp
)

file(GLOB FAISS_C_API_GPU_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
faiss_install_headers("${FAISS_C_API_GPU_HEADERS}" c_api/gpu)

if(FAISS_GPU_SEPARATE_LIBS)
  add_library(faiss_gpu_c ${FAISS_GPU_C_SRC})
  set_target_properties(faiss_gpu_c PROPERTIES OUTPUT_NAME faiss_${FAISS_GPU_BACKEND}_c)
  # Faiss headers contain OpenMP pragmas, see ../CMakeLists.txt
  target_link_libraries(faiss_gpu_c PRIVATE faiss_gpu faiss_c faiss OpenMP::OpenMP_CXX)
  target_link_options(faiss_gpu_c PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--no-undefined>)
  if (FAISS_ENABLE_ROCM)
    target_link_libraries(faiss_gpu_c PUBLIC hip::host roc::hipblas)
  else()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(faiss_gpu_c PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  endif()
  install(TARGETS faiss_gpu_c
    EXPORT faiss-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  set(FAISS_GPU_C faiss_gpu_c)
else()
  target_sources(faiss_c PRIVATE ${FAISS_GPU_C_SRC})
  if (FAISS_ENABLE_ROCM)
    target_link_libraries(faiss_c PUBLIC hip::host roc::hipblas)
    target_link_libraries(faiss_c_avx2 PUBLIC hip::host roc::hipblas)
    target_link_libraries(faiss_c_avx512 PUBLIC hip::host roc::hipblas)
    target_link_libraries(faiss_c_avx512_spr PUBLIC hip::host roc::hipblas)
    target_link_libraries(faiss_c_sve PUBLIC hip::host roc::hipblas)
  else()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(faiss_c PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(faiss_c_avx2 PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(faiss_c_avx512 PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(faiss_c_avx512_spr PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(faiss_c_sve PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  endif()
  set(FAISS_GPU_C faiss_c)
endif()

add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
target_link_libraries(example_gpu_c PRIVATE ${FAISS_GPU_C})
