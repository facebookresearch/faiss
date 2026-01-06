/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- C++ -*-

// This file contains GPU index-specific SWIG interface definitions
// It should be included AFTER the main CPU index headers are processed
// It is included from swigfaiss.swig when GPU_WRAPPER is defined

/*******************************************************************
 * GPU-specific vector templates
 *******************************************************************/

%template(GpuResourcesVector) std::vector<faiss::gpu::GpuResourcesProvider*>;

/*******************************************************************
 * GPU Index includes (requires CPU indices to be included first)
 *******************************************************************/

// quiet SWIG warnings
%ignore faiss::gpu::GpuIndexIVF::GpuIndexIVF;

%include  <faiss/gpu/GpuIndicesOptions.h>
%include  <faiss/gpu/GpuClonerOptions.h>
%include  <faiss/gpu/GpuIndex.h>
#ifdef FAISS_ENABLE_CUVS
%include  <faiss/gpu/GpuIndexCagra.h>
%include  <faiss/gpu/GpuIndexBinaryCagra.h>
#endif
%include  <faiss/gpu/GpuIndexFlat.h>
%include  <faiss/gpu/GpuIndexIVF.h>
%include  <faiss/gpu/GpuIndexIVFPQ.h>
%include  <faiss/gpu/GpuIndexIVFFlat.h>
%include  <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
%include  <faiss/gpu/GpuIndexBinaryFlat.h>
%include  <faiss/gpu/GpuDistance.h>
%include  <faiss/gpu/GpuIcmEncoder.h>

/*******************************************************************
 * GPU downcast support (Python-specific)
 *******************************************************************/

#ifdef SWIGPYTHON

%define DOWNCAST_GPU(subclass)
    if (dynamic_cast<faiss::gpu::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__gpu__ ## subclass,$owner);
    } else
%enddef

#endif // SWIGPYTHON

/*******************************************************************
 * GPU AutoTune and Cloner
 *******************************************************************/

%include  <faiss/gpu/GpuAutoTune.h>

%newobject index_gpu_to_cpu;
%newobject index_cpu_to_gpu;
%newobject index_cpu_to_gpu_multiple;

%include  <faiss/gpu/GpuCloner.h>
