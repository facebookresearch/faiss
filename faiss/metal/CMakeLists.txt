add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/kernels.metal -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib
  DEPENDS kernels.metal
)

add_library(faiss_metal_kernels INTERFACE)
target_link_libraries(faiss_metal_kernels INTERFACE "-framework Metal -framework Foundation")

add_library(faiss_metal SHARED
  MetalResources.mm
  MetalMemoryPool.mm
  MetalIndexFlat.mm
  MetalIndexIVFFlat.mm
  MetalIndexIVFPQ.mm
  MetalIndexHNSW.mm
  MetalMultiGpu.mm
  MetalMixedPrecision.mm
  MetalVectorTransform.mm
)

add_custom_target(faiss_metal_kernels_target DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib)
add_dependencies(faiss_metal faiss_metal_kernels_target)
target_link_libraries(faiss_metal PRIVATE faiss faiss_metal_kernels)
