/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- C++ -*-

// This file describes the C++-scripting language bridge for Python (and formerly Lua).
// It contains mainly includes and a few macros. There are
// 2 preprocessor macros of interest:

// SWIGPYTHON: Python-specific code
// GPU_WRAPPER: also compile interfaces for GPU.

%module swigfaiss;

// NOTE: While parsing the headers to generate the interface, SWIG does not know
// about `_MSC_VER`.
// TODO: Remove the need for this hack.
#ifdef SWIGWIN
#define _MSC_VER
%include <windows.i>
#endif // SWIGWIN

// fbode SWIG fails on warnings, so make them non fatal
#pragma SWIG nowarn=321
#pragma SWIG nowarn=403
#pragma SWIG nowarn=325
#pragma SWIG nowarn=389
#pragma SWIG nowarn=341
#pragma SWIG nowarn=512
#pragma SWIG nowarn=362
#pragma SWIG nowarn=509

// we need explict control of these typedefs...
// %include <stdint.i>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

// char != unsigned char AND char != signed char so be explicit
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


/*******************************************************************
 * Copied verbatim to wrapper. Contains the C++-visible includes, and
 * the language includes for their respective matrix libraries.
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

#include <faiss/impl/io.h>

#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/impl/mapped_io.h>
#include <faiss/impl/zerocopy_io.h>

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/Index2Layer.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFIndependentQuantizer.h>

#include <faiss/IndexFastScan.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/utils/simdlib.h>
#include <faiss/impl/simd_result_handlers.h>

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/utils/quantize_lut.h>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/impl/ThreadedIndex.h>
#include <faiss/IndexShards.h>
#include <faiss/IndexShardsIVF.h>
#include <faiss/IndexReplicas.h>
#include <faiss/impl/HNSW.h>
#include <faiss/IndexHNSW.h>

#include <faiss/impl/kmeans1d.h>

#include <faiss/impl/NNDescent.h>
#include <faiss/IndexNNDescent.h>

#include <faiss/impl/NSG.h>
#include <faiss/IndexNSG.h>

#include <faiss/MetaIndexes.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexRefine.h>

#include <faiss/IndexRowwiseMinMax.h>

#include <faiss/impl/FaissAssert.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryFromFloat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>

#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/clone_index.h>

#include <faiss/IVFlib.h>
#include <faiss/utils/utils.h>

#include <faiss/utils/sorting.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/partitioning.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/residual_quantizer_encode_steps.h>

#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductAdditiveQuantizer.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/utils/NeuralNet.h>

#include <faiss/invlists/BlockInvertedLists.h>

#ifndef _MSC_VER
#include <faiss/invlists/OnDiskInvertedLists.h>
#endif // !_MSC_VER

#include <faiss/Clustering.h>

#include <faiss/utils/hamming.h>
#include <faiss/utils/hamming_distance/common.h>

#include <faiss/AutoTune.h>
#include <faiss/MatrixStats.h>
#include <faiss/index_factory.h>

#include <faiss/impl/lattice_Zn.h>
#include <faiss/IndexLattice.h>

#include <faiss/IndexNeuralNetCodec.h>

#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexIVFRaBitQ.h>

%}

/********************************************************
 * GIL manipulation and exception handling
 ********************************************************/

#ifdef SWIGPYTHON
// %catches(faiss::FaissException);


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
    } catch(std::bad_alloc & ba) {
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
 * Types of vectors we want to manipulate at the scripting language
 * level.
 *******************************************************************/

// simplified interface for vector
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

%include <std_string.i>
%include <std_pair.i>
%include <std_map.i>
%include <std_shared_ptr.i>

// primitive array types
%template(Float32Vector) std::vector<float>;
%template(Float64Vector) std::vector<double>;

// weird interaction within C++ between char and signed char
%ignore Int8Vector::swap;

%template(Int8Vector) std::vector<int8_t>;
%template(Int16Vector) std::vector<int16_t>;
%template(Int32Vector) std::vector<int32_t>;
%template(Int64Vector) std::vector<int64_t>;

%template(UInt8Vector) std::vector<uint8_t>;
%template(UInt16Vector) std::vector<uint16_t>;
%template(UInt32Vector) std::vector<uint32_t>;
%template(UInt64Vector) std::vector<uint64_t>;

%template(Float32VectorVector) std::vector<std::vector<float> >;
%template(UInt8VectorVector) std::vector<std::vector<uint8_t> >;
%template(Int32VectorVector) std::vector<std::vector<int32_t> >;
%template(Int64VectorVector) std::vector<std::vector<int64_t> >;
%template(VectorTransformVector) std::vector<faiss::VectorTransform*>;
%template(OperatingPointVector) std::vector<faiss::OperatingPoint>;
%template(InvertedListsPtrVector) std::vector<faiss::InvertedLists*>;
%template(RepeatVector) std::vector<faiss::Repeat>;
%template(ClusteringIterationStatsVector) std::vector<faiss::ClusteringIterationStats>;
%template(ParameterRangeVector) std::vector<faiss::ParameterRange>;
%template(MaybeOwnedVectorUInt8Vector) std::vector<faiss::MaybeOwnedVector<uint8_t> >;
%template(MaybeOwnedVectorInt32Vector) std::vector<faiss::MaybeOwnedVector<int32_t> >;
%template(MaybeOwnedVectorFloat32Vector) std::vector<faiss::MaybeOwnedVector<float> >;

#ifndef SWIGWIN
%template(OnDiskOneListVector) std::vector<faiss::OnDiskOneList>;
#endif // !SWIGWIN

#ifdef GPU_WRAPPER
%template(GpuResourcesVector) std::vector<faiss::gpu::GpuResourcesProvider*>;
#endif

// produces an error on the Mac
%ignore faiss::hamming;

/*******************************************************************
 * Parse headers
 *******************************************************************/

%include <faiss/impl/platform_macros.h>

%ignore *::cmp;

%include <faiss/utils/ordered_key_value.h>
%include <faiss/utils/Heap.h>

// this ignore seems to be ignored, so disable W362 above
%ignore faiss::AlignedTable::operator=;

%include <faiss/utils/AlignedTable.h>
%include <faiss/utils/partitioning.h>
%include <faiss/utils/hamming.h>
%include <faiss/utils/hamming_distance/common.h>

int get_num_gpus();
void gpu_profiler_start();
void gpu_profiler_stop();
void gpu_sync_all_devices();

#ifdef GPU_WRAPPER
#ifdef FAISS_ENABLE_ROCM

%shared_ptr(faiss::gpu::GpuResources);
%shared_ptr(faiss::gpu::StandardGpuResourcesImpl);

%{

#include <faiss/gpu-rocm/StandardGpuResources.h>
#include <faiss/gpu-rocm/GpuIndicesOptions.h>
#include <faiss/gpu-rocm/GpuClonerOptions.h>
#include <faiss/gpu-rocm/GpuIndex.h>
#include <faiss/gpu-rocm/GpuIndexFlat.h>
#include <faiss/gpu-rocm/GpuIndexIVF.h>
#include <faiss/gpu-rocm/GpuIndexIVFPQ.h>
#include <faiss/gpu-rocm/GpuIndexIVFFlat.h>
#include <faiss/gpu-rocm/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu-rocm/GpuIndexBinaryFlat.h>
#include <faiss/gpu-rocm/GpuAutoTune.h>
#include <faiss/gpu-rocm/GpuCloner.h>
#include <faiss/gpu-rocm/GpuDistance.h>
#include <faiss/gpu-rocm/GpuIcmEncoder.h>

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

%include  <faiss/gpu-rocm/GpuResources.h>
%include  <faiss/gpu-rocm/StandardGpuResources.h>

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

%shared_ptr(faiss::gpu::GpuResources);
%shared_ptr(faiss::gpu::StandardGpuResourcesImpl);

%{

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>
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
#else // GPU_WRAPPER

%{
int get_num_gpus()
{
    return 0;
}

void gpu_profiler_start()
{
}

void gpu_profiler_stop()
{
}

void gpu_sync_all_devices()
{
}
%}


#endif // GPU_WRAPPER

// order matters because includes are not recursive

%include  <faiss/utils/utils.h>
%template(CombinerRangeKNNfloat) faiss::CombinerRangeKNN<float>;
%template(CombinerRangeKNNint16) faiss::CombinerRangeKNN<int16_t>;

%include  <faiss/utils/distances.h>
%include  <faiss/utils/random.h>
%include  <faiss/utils/sorting.h>

%include  <faiss/MetricType.h>

%newobject *::get_distance_computer() const;
%newobject *::get_CodePacker() const;

%include  <faiss/Index.h>

%include <faiss/impl/DistanceComputer.h>

%include <faiss/impl/io.h>

%ignore faiss::MmappedFileMappingOwner::p_impl;

%include <faiss/impl/maybe_owned_vector.h>
%include <faiss/impl/mapped_io.h>
%include <faiss/impl/zerocopy_io.h>

%newobject *::get_FlatCodesDistanceComputer() const;
%include  <faiss/IndexFlatCodes.h>
%include  <faiss/IndexFlat.h>
%include  <faiss/Clustering.h>

%include  <faiss/utils/extra_distances.h>

%ignore faiss::ProductQuantizer::get_centroids(size_t,size_t) const;

%include  <faiss/impl/Quantizer.h>
%include  <faiss/impl/ProductQuantizer.h>
%include  <faiss/impl/AdditiveQuantizer.h>
%include  <faiss/impl/residual_quantizer_encode_steps.h>
%include  <faiss/impl/ResidualQuantizer.h>
%include  <faiss/impl/LocalSearchQuantizer.h>
%include  <faiss/impl/ProductAdditiveQuantizer.h>
%include  <faiss/impl/CodePacker.h>

%include  <faiss/VectorTransform.h>
%include  <faiss/IndexPreTransform.h>
%include  <faiss/IndexRefine.h>
%include  <faiss/IndexLSH.h>
%include  <faiss/impl/PolysemousTraining.h>
%include  <faiss/IndexPQ.h>
%include  <faiss/IndexAdditiveQuantizer.h>
%include  <faiss/impl/io.h>

%include  <faiss/invlists/InvertedLists.h>
%include  <faiss/invlists/InvertedListsIOHook.h>
%ignore BlockInvertedListsIOHook;
%include  <faiss/invlists/BlockInvertedLists.h>
%include  <faiss/invlists/DirectMap.h>
%include  <faiss/IndexIVF.h>
// NOTE(hoss): SWIG (wrongly) believes the overloaded const version shadows the
//   non-const one.
%warnfilter(509) extract_index_ivf;
%warnfilter(509) try_extract_index_ivf;
%include  <faiss/IVFlib.h>
%include  <faiss/impl/ScalarQuantizer.h>
%include  <faiss/IndexScalarQuantizer.h>
%include  <faiss/IndexIVFSpectralHash.h>
%include  <faiss/IndexIVFAdditiveQuantizer.h>
%include  <faiss/impl/HNSW.h>
%include  <faiss/IndexHNSW.h>

%include <faiss/impl/kmeans1d.h>

%ignore faiss::nndescent::Nhood::lock;
%include  <faiss/impl/NNDescent.h>
%include  <faiss/IndexNNDescent.h>

%include  <faiss/IndexIVFFlat.h>

%warnfilter(509) faiss::nsg::Graph< int >::at(int,int);

%include  <faiss/impl/NSG.h>

%template(NSG_Graph_int) faiss::nsg::Graph<int>;

// not using %shared_ptr to avoid mem leaks
%extend faiss::NSG {
    faiss::nsg::Graph<int>* get_final_graph() {
        return $self->final_graph.get();
    }
}

%include  <faiss/IndexNSG.h>

#ifndef SWIGWIN
%warnfilter(401) faiss::OnDiskInvertedListsIOHook;
%ignore OnDiskInvertedListsIOHook;
%include  <faiss/invlists/OnDiskInvertedLists.h>
#endif // !SWIGWIN

%include  <faiss/impl/lattice_Zn.h>
%include  <faiss/IndexLattice.h>

%ignore faiss::IndexIVFPQ::alloc_type;
%include  <faiss/IndexIVFPQ.h>
%include  <faiss/IndexIVFPQR.h>
%include  <faiss/Index2Layer.h>

%include  <faiss/IndexFastScan.h>
%include  <faiss/IndexAdditiveQuantizerFastScan.h>
%include  <faiss/IndexPQFastScan.h>

// NOTE(matthijs) let's not go into wrapping simdlib
struct faiss::simd16uint16 {};

%include  <faiss/impl/simd_result_handlers.h>
%include  <faiss/IndexIVFFastScan.h>
%include  <faiss/IndexIVFAdditiveQuantizerFastScan.h>
%include  <faiss/IndexIVFIndependentQuantizer.h>

%include  <faiss/IndexIVFPQFastScan.h>
%include  <faiss/utils/quantize_lut.h>

%include  <faiss/IndexBinary.h>
%include  <faiss/IndexBinaryFlat.h>
%include  <faiss/IndexBinaryIVF.h>
%include  <faiss/IndexBinaryFromFloat.h>
%include  <faiss/IndexBinaryHNSW.h>
%include  <faiss/IndexBinaryHash.h>

%include  <faiss/impl/ThreadedIndex.h>
%template(ThreadedIndexBase) faiss::ThreadedIndex<faiss::Index>;
%template(ThreadedIndexBaseBinary) faiss::ThreadedIndex<faiss::IndexBinary>;

%include  <faiss/IndexShards.h>
%template(IndexShards) faiss::IndexShardsTemplate<faiss::Index>;
%template(IndexBinaryShards) faiss::IndexShardsTemplate<faiss::IndexBinary>;
%include  <faiss/IndexShardsIVF.h>

%include  <faiss/IndexReplicas.h>
%template(IndexReplicas) faiss::IndexReplicasTemplate<faiss::Index>;
%template(IndexBinaryReplicas) faiss::IndexReplicasTemplate<faiss::IndexBinary>;

%include  <faiss/MetaIndexes.h>

%include  <faiss/IndexRowwiseMinMax.h>

%include <faiss/utils/NeuralNet.h>
%template(Tensor2D) faiss::nn::Tensor2DTemplate<float>;
%template(Int32Tensor2D) faiss::nn::Tensor2DTemplate<int32_t>;

%include <faiss/IndexNeuralNetCodec.h>

%include <faiss/impl/RaBitQuantizer.h>
%include <faiss/IndexRaBitQ.h>
%include <faiss/IndexIVFRaBitQ.h>

%ignore faiss::BufferList::Buffer;
%ignore faiss::RangeSearchPartialResult::QueryResult;
%ignore faiss::IDSelectorBatch::set;
%ignore faiss::IDSelectorBatch::bloom;
%ignore faiss::InterruptCallback::instance;
%ignore faiss::InterruptCallback::lock;

%include <faiss/impl/AuxIndexStructures.h>
%include <faiss/impl/IDSelector.h>

%include  <faiss/IndexIDMap.h>
%template(IndexIDMap) faiss::IndexIDMapTemplate<faiss::Index>;
%template(IndexBinaryIDMap) faiss::IndexIDMapTemplate<faiss::IndexBinary>;
%template(IndexIDMap2) faiss::IndexIDMap2Template<faiss::Index>;
%template(IndexBinaryIDMap2) faiss::IndexIDMap2Template<faiss::IndexBinary>;


%include <faiss/utils/approx_topk/mode.h>

#ifdef GPU_WRAPPER

#ifdef FAISS_ENABLE_ROCM

// quiet SWIG warnings
%ignore faiss::gpu::GpuIndexIVF::GpuIndexIVF;

%include  <faiss/gpu-rocm/GpuIndicesOptions.h>
%include  <faiss/gpu-rocm/GpuClonerOptions.h>
%include  <faiss/gpu-rocm/GpuIndex.h>
%include  <faiss/gpu-rocm/GpuIndexFlat.h>
%include  <faiss/gpu-rocm/GpuIndexIVF.h>
%include  <faiss/gpu-rocm/GpuIndexIVFPQ.h>
%include  <faiss/gpu-rocm/GpuIndexIVFFlat.h>
%include  <faiss/gpu-rocm/GpuIndexIVFScalarQuantizer.h>
%include  <faiss/gpu-rocm/GpuIndexBinaryFlat.h>
%include  <faiss/gpu-rocm/GpuDistance.h>
%include  <faiss/gpu-rocm/GpuIcmEncoder.h>

#else // FAISS_ENABLE_ROCM

// quiet SWIG warnings
%ignore faiss::gpu::GpuIndexIVF::GpuIndexIVF;

%include  <faiss/gpu/GpuIndicesOptions.h>
%include  <faiss/gpu/GpuClonerOptions.h>
%include  <faiss/gpu/GpuIndex.h>
#ifdef FAISS_ENABLE_CUVS
%include  <faiss/gpu/GpuIndexCagra.h>
#endif
%include  <faiss/gpu/GpuIndexFlat.h>
%include  <faiss/gpu/GpuIndexIVF.h>
%include  <faiss/gpu/GpuIndexIVFPQ.h>
%include  <faiss/gpu/GpuIndexIVFFlat.h>
%include  <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
%include  <faiss/gpu/GpuIndexBinaryFlat.h>
%include  <faiss/gpu/GpuDistance.h>
%include  <faiss/gpu/GpuIcmEncoder.h>

#endif // FAISS_ENABLE_ROCM
#endif






/*******************************************************************
 * downcast return of some functions so that the sub-class is used
 * instead of the generic upper-class.
 *******************************************************************/


#ifdef SWIGPYTHON

%define DOWNCAST(subclass)
    if (dynamic_cast<faiss::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## subclass,$owner);
    } else
%enddef

%define DOWNCAST2(subclass, longname)
    if (dynamic_cast<faiss::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__ ## longname,$owner);
    } else
%enddef

%define DOWNCAST_GPU(subclass)
    if (dynamic_cast<faiss::gpu::subclass *> ($1)) {
      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__gpu__ ## subclass,$owner);
    } else
%enddef

#endif

%newobject read_index;
%newobject read_index_binary;
%newobject read_VectorTransform;
%newobject read_ProductQuantizer;
%newobject clone_index;
%newobject clone_binary_index;
%newobject clone_Quantizer;
%newobject clone_VectorTransform;

// Subclasses should appear before their parent
%typemap(out) faiss::Index * {
    DOWNCAST2 ( IndexIDMap2, IndexIDMap2TemplateT_faiss__Index_t )
    DOWNCAST2 ( IndexIDMap, IndexIDMapTemplateT_faiss__Index_t )
    DOWNCAST ( IndexShardsIVF )
    DOWNCAST2 ( IndexShards, IndexShardsTemplateT_faiss__Index_t )
    DOWNCAST2 ( IndexReplicas, IndexReplicasTemplateT_faiss__Index_t )
    DOWNCAST ( IndexRaBitQ )
    DOWNCAST ( IndexIVFRaBitQ )
    DOWNCAST ( IndexIVFIndependentQuantizer)
    DOWNCAST ( IndexIVFPQR )
    DOWNCAST ( IndexIVFPQ )
    DOWNCAST ( IndexIVFPQFastScan )
    DOWNCAST ( IndexIVFSpectralHash )
    DOWNCAST ( IndexIVFScalarQuantizer )
    DOWNCAST ( IndexIVFResidualQuantizer )
    DOWNCAST ( IndexIVFLocalSearchQuantizer )
    DOWNCAST ( IndexIVFProductResidualQuantizer )
    DOWNCAST ( IndexIVFProductLocalSearchQuantizer )
    DOWNCAST ( IndexIVFResidualQuantizerFastScan )
    DOWNCAST ( IndexIVFLocalSearchQuantizerFastScan )
    DOWNCAST ( IndexIVFProductResidualQuantizerFastScan )
    DOWNCAST ( IndexIVFProductLocalSearchQuantizerFastScan )
    DOWNCAST ( IndexIVFFlatDedup )
    DOWNCAST ( IndexIVFFlat )
    DOWNCAST ( IndexIVF )
    DOWNCAST ( IndexFlatIP )
    DOWNCAST ( IndexFlatL2 )
    DOWNCAST ( IndexFlat )
    DOWNCAST ( IndexRefineFlat )
    DOWNCAST ( IndexRefine )
    DOWNCAST ( IndexPQFastScan )
    DOWNCAST ( IndexPQ )
    DOWNCAST ( IndexResidualQuantizer )
    DOWNCAST ( IndexLocalSearchQuantizer )
    DOWNCAST ( IndexResidualQuantizerFastScan )
    DOWNCAST ( IndexLocalSearchQuantizerFastScan )
    DOWNCAST ( IndexProductResidualQuantizerFastScan )
    DOWNCAST ( IndexProductLocalSearchQuantizerFastScan )
    DOWNCAST ( ResidualCoarseQuantizer )
    DOWNCAST ( LocalSearchCoarseQuantizer )
    DOWNCAST ( IndexProductResidualQuantizer )
    DOWNCAST ( IndexProductLocalSearchQuantizer )
    DOWNCAST ( IndexScalarQuantizer )
    DOWNCAST ( IndexLSH )
    DOWNCAST ( IndexLattice )
    DOWNCAST ( IndexPreTransform )
    DOWNCAST ( MultiIndexQuantizer )
    DOWNCAST ( IndexHNSWFlat )
    DOWNCAST ( IndexHNSWPQ )
    DOWNCAST ( IndexHNSWSQ )
    DOWNCAST ( IndexHNSW )
    DOWNCAST ( IndexHNSW2Level )
    DOWNCAST ( IndexNNDescentFlat )
    DOWNCAST ( IndexNSGFlat )
    DOWNCAST ( IndexNSGPQ )
    DOWNCAST ( IndexNSGSQ )
    DOWNCAST ( Index2Layer )
    DOWNCAST ( IndexRandom )
    DOWNCAST ( IndexRowwiseMinMax )
    DOWNCAST ( IndexRowwiseMinMaxFP16 )
#ifdef GPU_WRAPPER
#ifdef FAISS_ENABLE_CUVS
    DOWNCAST_GPU ( GpuIndexCagra )
#endif
    DOWNCAST_GPU ( GpuIndexIVFPQ )
    DOWNCAST_GPU ( GpuIndexIVFFlat )
    DOWNCAST_GPU ( GpuIndexIVFScalarQuantizer )
    DOWNCAST_GPU ( GpuIndexFlat )
#endif
    // default for non-recognized classes
    DOWNCAST ( Index )
    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}


%typemap(out) faiss::IndexBinary * {
    DOWNCAST2 ( IndexBinaryReplicas, IndexReplicasTemplateT_faiss__IndexBinary_t )
    DOWNCAST2 ( IndexBinaryIDMap2, IndexIDMap2TemplateT_faiss__IndexBinary_t )
    DOWNCAST2 ( IndexBinaryIDMap, IndexIDMapTemplateT_faiss__IndexBinary_t )
    DOWNCAST ( IndexBinaryIVF )
    DOWNCAST ( IndexBinaryFlat )
    DOWNCAST ( IndexBinaryFromFloat )
    DOWNCAST ( IndexBinaryHNSW )
    DOWNCAST ( IndexBinaryHash )
    DOWNCAST ( IndexBinaryMultiHash )
#ifdef GPU_WRAPPER
    DOWNCAST_GPU ( GpuIndexBinaryFlat )
#endif
    // default for non-recognized classes
    DOWNCAST ( IndexBinary )
    if ($1 == NULL)
    {
#ifdef SWIGPYTHON
        $result = SWIG_Py_Void();
#endif
    } else {
        assert(false);
    }
}

%typemap(out) faiss::VectorTransform * {
    DOWNCAST (RemapDimensionsTransform)
    DOWNCAST (OPQMatrix)
    DOWNCAST (PCAMatrix)
    DOWNCAST (ITQMatrix)
    DOWNCAST (RandomRotationMatrix)
    DOWNCAST (LinearTransform)
    DOWNCAST (NormalizationTransform)
    DOWNCAST (CenteringTransform)
    DOWNCAST (ITQTransform)
    DOWNCAST (VectorTransform)
    {
        assert(false);
    }
}

%typemap(out) faiss::InvertedLists * {
    DOWNCAST (ArrayInvertedLists)
    DOWNCAST (BlockInvertedLists)
#ifndef SWIGWIN
    DOWNCAST (OnDiskInvertedLists)
#endif // !SWIGWIN
    DOWNCAST (VStackInvertedLists)
    DOWNCAST (HStackInvertedLists)
    DOWNCAST (MaskedInvertedLists)
    DOWNCAST (InvertedLists)
    {
        assert(false);
    }
}

%typemap(out) faiss::Quantizer * {
    DOWNCAST (ScalarQuantizer)
    DOWNCAST (ProductQuantizer)
    DOWNCAST (LocalSearchQuantizer)
    DOWNCAST (ResidualQuantizer)
    DOWNCAST (ProductLocalSearchQuantizer)
    DOWNCAST (ProductResidualQuantizer)
    {
        assert(false);
    }
}

// just to downcast pointers that come from elsewhere (eg. direct
// access to object fields)
%inline %{
faiss::Index * downcast_index (faiss::Index *index)
{
    return index;
}
faiss::VectorTransform * downcast_VectorTransform (faiss::VectorTransform *vt)
{
    return vt;
}
faiss::IndexBinary * downcast_IndexBinary (faiss::IndexBinary *index)
{
    return index;
}
faiss::InvertedLists * downcast_InvertedLists (faiss::InvertedLists *il)
{
    return il;
}
// backwards compatibility
faiss::Quantizer * downcast_AdditiveQuantizer (faiss::AdditiveQuantizer *aq)
{
    return aq;
}
faiss::Quantizer * downcast_Quantizer (faiss::Quantizer *aq)
{
    return aq;
}
%}

%include  <faiss/index_io.h>
%include  <faiss/clone_index.h>
%newobject index_factory;
%newobject index_binary_factory;

%include  <faiss/AutoTune.h>
%include  <faiss/index_factory.h>
%include  <faiss/MatrixStats.h>


#ifdef GPU_WRAPPER

#ifdef FAISS_ENABLE_ROCM
%include  <faiss/gpu-rocm/GpuAutoTune.h>

%newobject index_gpu_to_cpu;
%newobject index_cpu_to_gpu;
%newobject index_cpu_to_gpu_multiple;

%include  <faiss/gpu-rocm/GpuCloner.h>

#else // FAISS_ENABLE_ROCM
%include  <faiss/gpu/GpuAutoTune.h>

%newobject index_gpu_to_cpu;
%newobject index_cpu_to_gpu;
%newobject index_cpu_to_gpu_multiple;

%include  <faiss/gpu/GpuCloner.h>

#endif // FAISS_ENABLE_ROCM
#endif



/*******************************************************************
 * Support I/O to arbitrary functions
 *******************************************************************/


#ifdef SWIGPYTHON
%include <faiss/python/python_callbacks.h>


%{
#include <faiss/python/python_callbacks.h>
%}

#endif


/*******************************************************************
 * How should the template objects appear in the scripting language?
 *******************************************************************/

// answer: the same as the C++ typedefs, but we still have to redefine them

%template() faiss::CMin<float, int64_t>;
%template() faiss::CMin<int, int64_t>;
%template() faiss::CMax<float, int64_t>;
%template() faiss::CMax<int, int64_t>;

%template(float_minheap_array_t) faiss::HeapArray<faiss::CMin<float, int64_t> >;
%template(int_minheap_array_t) faiss::HeapArray<faiss::CMin<int, int64_t> >;
%template(float_maxheap_array_t) faiss::HeapArray<faiss::CMax<float, int64_t> >;
%template(int_maxheap_array_t) faiss::HeapArray<faiss::CMax<int, int64_t> >;

%template(CMin_float_partition_fuzzy)
    faiss::partition_fuzzy<faiss::CMin<float, int64_t> >;
%template(CMax_float_partition_fuzzy)
    faiss::partition_fuzzy<faiss::CMax<float, int64_t> >;

%template(AlignedTableUint8) faiss::AlignedTable<uint8_t>;
%template(AlignedTableUint16) faiss::AlignedTable<uint16_t>;
%template(AlignedTableFloat32) faiss::AlignedTable<float>;

%template(MaybeOwnedVectorUInt8) faiss::MaybeOwnedVector<uint8_t>;
%template(MaybeOwnedVectorInt32) faiss::MaybeOwnedVector<int32_t>;
%template(MaybeOwnedVectorFloat32) faiss::MaybeOwnedVector<float>;


// SWIG seems to have some trouble resolving function template types here, so
// declare explicitly

%define INSTANTIATE_uint16_partition_fuzzy(C, id_t)

%inline %{

uint16_t C ## _uint16_partition_fuzzy(
        uint16_t *vals, id_t *ids, size_t n,
        size_t q_min, size_t q_max, size_t * q_out)
{
    return faiss::partition_fuzzy<faiss::C<unsigned short, id_t> >(
        vals, ids, n, q_min, q_max, q_out);
}

%}

%enddef

INSTANTIATE_uint16_partition_fuzzy(CMin, int64_t)
INSTANTIATE_uint16_partition_fuzzy(CMax, int64_t)
INSTANTIATE_uint16_partition_fuzzy(CMin, int)
INSTANTIATE_uint16_partition_fuzzy(CMax, int)

// Same for merge_knn_results

// same define as explicit instanciation in Heap.cpp
%define INSTANTIATE_merge_knn_results(C, distance_t)

%inline %{
void merge_knn_results_ ## C(
    size_t n, size_t k, int nshard,
    const distance_t *all_distances, const faiss::idx_t *all_labels,
    distance_t *distances, faiss::idx_t *labels)
{
    faiss::merge_knn_results<faiss::idx_t, faiss::C<distance_t, int>>(
        n, k, nshard, all_distances, all_labels, distances, labels);
}
%}

%enddef

INSTANTIATE_merge_knn_results(CMin, float);
INSTANTIATE_merge_knn_results(CMax, float);
INSTANTIATE_merge_knn_results(CMin, int32_t);
INSTANTIATE_merge_knn_results(CMax, int32_t);



%inline %{

/*******************************************************************
 * numpy misses a hash table implementation, hence this class. It
 * represents not found values as -1 like in the Index implementation
 *******************************************************************/


struct MapLong2Long {
    std::unordered_map<int64_t, int64_t> map;

    void add(size_t n, const int64_t *keys, const int64_t *vals) {
        map.reserve(map.size() + n);
        for (size_t i = 0; i < n; i++) {
            map[keys[i]] = vals[i];
        }
    }

    int64_t search(int64_t key) const {
        auto ptr = map.find(key);
        if (ptr == map.end()) {
            return -1;
        } else {
            return ptr->second;
        }
    }

    void search_multiple(size_t n, int64_t *keys, int64_t * vals) {
        for (size_t i = 0; i < n; i++) {
            vals[i] = search(keys[i]);
        }
    }
};

%}


/*******************************************************************
 * Expose a few basic functions
 *******************************************************************/


void omp_set_num_threads (int num_threads);
int omp_get_max_threads ();
void *memcpy(void *dest, const void *src, size_t n);




/*******************************************************************
 * Python-specific: do not release GIL any more, as functions below
 * use the Python/C API
 *******************************************************************/

#ifdef SWIGPYTHON
%exception;
#endif



/*******************************************************************
 * Python specific: numpy array <-> C++ pointer interface
 *******************************************************************/

#ifdef SWIGPYTHON

// transfer SWIG flag to C++
#ifdef SWIGWORDSIZE64
%{
#define SWIGWORDSIZE64_CPP
%}
#endif

%{
PyObject *swig_ptr (PyObject *a)
{

    if (PyBytes_Check(a)) {
        return SWIG_NewPointerObj(PyBytes_AsString(a), SWIGTYPE_p_char, 0);
    }
    if (PyByteArray_Check(a)) {
        return SWIG_NewPointerObj(PyByteArray_AsString(a), SWIGTYPE_p_char, 0);
    }
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_signed_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_BOOL) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_bool, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
    // Convert npy64 either long or long long  and it depends on how compiler define int64_t.
    // In the 64bit machine, typically the int64_t should be long but it is not hold for Apple osx.
    // In this case, we want to convert npy64 to long_Long in osx
#ifdef SWIGWORDSIZE64_CPP
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long_long, 0);
#endif
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
#ifdef SWIGWORDSIZE64_CPP
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long_long, 0);
#endif
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}
%}

%inline %{

struct PythonInterruptCallback: faiss::InterruptCallback {

    bool want_interrupt () override {
        int err;
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            err = PyErr_CheckSignals();
            PyGILState_Release(gstate);
        }
        return err == -1;
    }

    static void reset() {
        faiss::InterruptCallback::instance.reset(new PythonInterruptCallback());
    }
};

%}

%init %{
    /* needed, else crash at runtime */
    import_array();

    PythonInterruptCallback::reset();
%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)

%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef

REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(double, NPY_FLOAT64);
REV_SWIG_PTR(uint8_t, NPY_UINT8);
REV_SWIG_PTR(int8_t, NPY_INT8);
REV_SWIG_PTR(unsigned short, NPY_UINT16);
REV_SWIG_PTR(short, NPY_INT16);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(unsigned int, NPY_UINT32);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);

#endif



/*******************************************************************
 * For Faiss/Pytorch interop via pointers encoded as longs
 *******************************************************************/

%inline %{
uint8_t * cast_integer_to_uint8_ptr (int64_t x) {
    return (uint8_t*)x;
}

float * cast_integer_to_float_ptr (int64_t x) {
    return (float*)x;
}

faiss::idx_t* cast_integer_to_idx_t_ptr (int64_t x) {
    return (faiss::idx_t*)x;
}

int * cast_integer_to_int_ptr (int64_t x) {
    return (int*)x;
}

void * cast_integer_to_void_ptr (int64_t x) {
    return (void*)x;
}
%}

%inline %{

// the SWIG version is a 6-digit hex string, eg. version 3.2.1 is encoded as
// 0x030201
uint64_t swig_version() {
    return SWIG_VERSION;
}

%}

// End of file...
