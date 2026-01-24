/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- C++ -*-

// This is a separate GPU module that can be compiled independently from
// the main CPU faiss module. It imports types from the CPU module and
// provides GPU-specific functionality.

%module swigfaiss_gpu_module;

// Type definitions needed by SWIG
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;

#ifdef SWIGWORDSIZE64
typedef unsigned long uint64_t;
typedef long int64_t;
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

typedef uint64_t size_t;

#define __restrict
#define ALIGNED(x)
#define FAISS_API

/*******************************************************************
 * GPU module header includes
 *******************************************************************/

%{

#include <stdint.h>
#include <omp.h>

#ifdef SWIGPYTHON

#undef popcount64

#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#endif

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

/********************************************************
 * GIL manipulation and exception handling
 ********************************************************/

#ifdef SWIGPYTHON

// Python-specific: release GIL by default for all functions
%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(faiss::FaissException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc&) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    } catch(const std::exception& ex) {
        PyEval_RestoreThread(_save);
        std::string what = std::string("C++ exception ") + ex.what();
        PyErr_SetString(PyExc_RuntimeError, what.c_str());
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}

#endif

/*******************************************************************
 * Import types and classes from the main faiss module
 *******************************************************************/

%import(module="faiss") <std_string.i>
%import(module="faiss") <std_pair.i>
%import(module="faiss") <std_map.i>
%import(module="faiss") <std_shared_ptr.i>

%import(module="faiss") "faiss/MetricType.h"
%import(module="faiss") "faiss/Index.h"
%import(module="faiss") "faiss/IndexBinary.h"
%import(module="faiss") "faiss/IndexIVF.h"
%import(module="faiss") "faiss/impl/AuxIndexStructures.h"
%import(module="faiss") "faiss/impl/FaissException.h"
%import(module="faiss") "faiss/impl/LocalSearchQuantizer.h"
%import(module="faiss") "faiss/impl/ProductQuantizer.h"
%import(module="faiss") "faiss/impl/AdditiveQuantizer.h"
%import(module="faiss") "faiss/Clustering.h"
%import(module="faiss") "faiss/AutoTune.h"
%import(module="faiss") "faiss/clone_index.h"

/*******************************************************************
 * GPU-specific declarations
 *******************************************************************/

// GPU utility functions (defined in %{ %} above)
int get_num_gpus();
void gpu_profiler_start();
void gpu_profiler_stop();
void gpu_sync_all_devices();

// Vector templates needed for GPU
namespace std {
    template<class T>
    class vector {
    public:
        vector();
        void push_back(T);
        void clear();
        T * data();
        size_t size();
        T at (size_t n) const;
        T & operator [] (size_t n);
        void resize (size_t n);
        void swap (vector<T> & other);
    };
};

%template(GpuResourcesVector) std::vector<faiss::gpu::GpuResourcesProvider*>;

%shared_ptr(faiss::gpu::GpuResources);
%shared_ptr(faiss::gpu::StandardGpuResourcesImpl);

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

/*******************************************************************
 * GPU Index includes
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
 * GPU AutoTune and Cloner
 *******************************************************************/

%include  <faiss/gpu/GpuAutoTune.h>

%newobject index_gpu_to_cpu;
%newobject index_cpu_to_gpu;
%newobject index_cpu_to_gpu_multiple;

%include  <faiss/gpu/GpuCloner.h>

/*******************************************************************
 * Python-specific: numpy array handling
 *******************************************************************/

#ifdef SWIGPYTHON

%init %{
    /* needed, else crash at runtime */
    import_array();
%}

#endif
