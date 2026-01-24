/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- C++ -*-

// This file contains GPU-specific SWIG interface definitions for basic GPU setup
// It is included early from swigfaiss.swig when GPU_WRAPPER is defined
// GPU index includes are in swigfaiss_gpu_indices.swig (included later)

/*******************************************************************
 * GPU-specific resources and functions
 *******************************************************************/

%shared_ptr(faiss::gpu::GpuResources);
%shared_ptr(faiss::gpu::StandardGpuResourcesImpl);

%{

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIcmEncoder.h>

int get_num_gpus()
{
    return faiss::gpu::getNumDevices();
}

void gpu_profiler_start()
{
    return faiss::gpu::profilerStart();
}

void gpu_profiler_stop()
{
    return faiss::gpu::profilerStop();
}

void gpu_sync_all_devices()
{
    return faiss::gpu::synchronizeAllDevices();
}

%}

%template() std::pair<int, uint64_t>;
%template() std::map<std::string, std::pair<int, uint64_t> >;
%template() std::map<int, std::map<std::string, std::pair<int, uint64_t> > >;

// causes weird wrapper bug
%ignore *::allocMemoryHandle;
%ignore faiss::gpu::GpuMemoryReservation;
%ignore faiss::gpu::GpuMemoryReservation::operator=(GpuMemoryReservation&&);
%ignore faiss::gpu::AllocType;

%include  <faiss/gpu/GpuResources.h>
%include  <faiss/gpu/StandardGpuResources.h>

#ifdef FAISS_ENABLE_ROCM

typedef ihipStream_t* hipStream_t;

%inline %{

// interop between pytorch exposed hipStream_t and faiss
hipStream_t cast_integer_to_cudastream_t(int64_t x) {
  return (hipStream_t) x;
}

int64_t cast_cudastream_t_to_integer(hipStream_t x) {
  return (int64_t) x;
}

%}

#else // FAISS_ENABLE_ROCM

#ifdef FAISS_ENABLE_CUVS
%shared_ptr(faiss::gpu::IVFPQBuildCagraConfig);
%shared_ptr(faiss::gpu::IVFPQSearchCagraConfig);

%{

#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexBinaryCagra.h>

%}
#endif // FAISS_ENABLE_CUVS

typedef CUstream_st* cudaStream_t;

%inline %{

// interop between pytorch exposed cudaStream_t and faiss
cudaStream_t cast_integer_to_cudastream_t(int64_t x) {
  return (cudaStream_t) x;
}

int64_t cast_cudastream_t_to_integer(cudaStream_t x) {
  return (int64_t) x;
}

%}

#endif // FAISS_ENABLE_ROCM
