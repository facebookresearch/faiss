# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

set(FAISS_GPU_SRC 
  utils/DeviceUtils.cu

)

set(FAISS_GPU_HEADERS 
  utils/DeviceDefs.cuh
  utils/DeviceUtils.h
)

function(generate_ivf_interleaved_code)
  set(SUB_CODEC_TYPE
    "faiss::gpu::Codec<0, 1>"
    "faiss::gpu::Codec<1, 1>"
    "faiss::gpu::Codec<2, 1>"
    "faiss::gpu::Codec<3, 1>"
    "faiss::gpu::Codec<4, 1>"
    "faiss::gpu::Codec<5, 1>"
    "faiss::gpu::Codec<6, 1>"
    "faiss::gpu::CodecFloat"
  )

  set(SUB_METRIC_TYPE
    "faiss::gpu::IPDistance"
    "faiss::gpu::L2Distance"
  )

  # Used for SUB_THREADS, SUB_NUM_WARP_Q, SUB_NUM_THREAD_Q
  set(THREADS_AND_WARPS
    "128|1024|8"
    "128|1|1"
    "128|128|3"
    "128|256|4"
    "128|32|2"
    "128|512|8"
    "128|64|3"
    "64|2048|8"
  )

  # Traverse through the Cartesian product of X and Y
  foreach(sub_codec ${SUB_CODEC_TYPE})
  break()
  foreach(metric_type ${SUB_METRIC_TYPE})
  foreach(threads_and_warps_str ${THREADS_AND_WARPS})
    string(REPLACE "|" ";" threads_and_warps ${threads_and_warps_str})
    list(GET threads_and_warps 0 sub_threads)
    list(GET threads_and_warps 1 sub_num_warp_q)
    list(GET threads_and_warps 2 sub_num_thread_q)

    # Define the output file name
    set(filename "template_${sub_codec}_${metric_type}_${sub_threads}_${sub_num_warp_q}_${sub_num_thread_q}")
    # Remove illegal characters from filename
    string(REGEX REPLACE "[^A-Za-z0-9_]" "" filename ${filename})
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${filename}.cu")

    # Read the template file
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/impl/scan/IVFInterleavedScanKernelTemplate.cu" template_content)

    # Replace the placeholders
    string(REPLACE "SUB_CODEC_TYPE" "${sub_codec}" template_content "${template_content}")
    string(REPLACE "SUB_METRIC_TYPE" "${metric_type}" template_content "${template_content}")
    string(REPLACE "SUB_THREADS" "${sub_threads}" template_content "${template_content}")
    string(REPLACE "SUB_NUM_WARP_Q" "${sub_num_warp_q}" template_content "${template_content}")
    string(REPLACE "SUB_NUM_THREAD_Q" "${sub_num_thread_q}" template_content "${template_content}")

    # Write the modified content to the output file
    file(WRITE "${output_file}" "${template_content}")

    # Add the file to the sources
    list(APPEND FAISS_GPU_SRC "${output_file}")
  endforeach()
  endforeach()
  endforeach()
  # Propagate modified variable to the parent scope
  set(FAISS_GPU_SRC "${FAISS_GPU_SRC}" PARENT_SCOPE)
endfunction()

generate_ivf_interleaved_code()

#if(FAISS_ENABLE_RAFT)
#  list(APPEND FAISS_GPU_HEADERS
#          GpuIndexCagra.h
#          impl/RaftCagra.cuh
#         impl/RaftFlatIndex.cuh
#          impl/RaftIVFFlat.cuh
#          impl/RaftIVFPQ.cuh
#          utils/RaftUtils.h)
#  list(APPEND FAISS_GPU_SRC
#          GpuIndexCagra.cu
#          impl/RaftCagra.cu
#          impl/RaftFlatIndex.cu
#          impl/RaftIVFFlat.cu
#          impl/RaftIVFPQ.cu
#          utils/RaftUtils.cu)
#endif()

add_library(faiss_gpu STATIC ${FAISS_GPU_SRC})
set_target_properties(faiss_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_include_directories(faiss_gpu PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

if(FAISS_ENABLE_RAFT)
  target_compile_definitions(faiss PUBLIC USE_NVIDIA_RAFT=1)
  target_compile_definitions(faiss_avx2 PUBLIC USE_NVIDIA_RAFT=1)
  target_compile_definitions(faiss_avx512 PUBLIC USE_NVIDIA_RAFT=1)

  # Mark all functions as hidden so that we don't generate
  # global 'public' functions that also exist in libraft.so
  #
  # This ensures that faiss functions will call the local version
  # inside libfaiss.so . This is needed to ensure that things
  # like raft cublas resources are created and used within the same
  # dynamic library + CUDA runtime context which are requirements
  # for valid execution
  #
  # To still allow these classes to be used by consumers, the
  # respective classes/types in the headers are explicitly marked
  # as 'public' so they can be used by consumers
  set_source_files_properties(
    GpuDistance.cu
    StandardGpuResources.cpp
    impl/RaftFlatIndex.cu
    impl/RaftIVFFlat.cu
    impl/RaftIVFPQ.cu
    utils/RaftUtils.cu
    TARGET_DIRECTORY faiss
    PROPERTIES COMPILE_OPTIONS "-fvisibility=hidden")
  target_compile_definitions(faiss_gpu PUBLIC USE_NVIDIA_RAFT=1)
endif()

# Export FAISS_GPU_HEADERS variable to parent scope.
set(FAISS_GPU_HEADERS ${FAISS_GPU_HEADERS} PARENT_SCOPE)

target_link_libraries(faiss PRIVATE  "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
target_link_libraries(faiss_avx2 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
target_link_libraries(faiss_avx512 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")

foreach(header ${FAISS_GPU_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/${dir}
  )
endforeach()

# Prepares a host linker script and enables host linker to support
# very large device object files.
# This is what CUDA 11.5+ `nvcc -hls=gen-lcs -aug-hls` would generate
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
__nv_relfatbin : { *(__nv_relfatbin) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
)
target_link_options(faiss_gpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(faiss_gpu PRIVATE CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft> $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::compiled> $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass> $<$<BOOL:${FAISS_ENABLE_RAFT}>:OpenMP::OpenMP_CXX>)
target_compile_options(faiss_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all --expt-extended-lambda --expt-relaxed-constexpr $<$<BOOL:${FAISS_ENABLE_RAFT}>:-Xcompiler=${OpenMP_CXX_FLAGS}>>)
