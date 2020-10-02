/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>

namespace faiss { namespace gpu {

IVFFlat::IVFFlat(GpuResources* res,
                 FlatIndex* quantizer,
                 faiss::MetricType metric,
                 float metricArg,
                 bool useResidual,
                 faiss::ScalarQuantizer* scalarQ,
                 IndicesOptions indicesOptions,
                 MemorySpace space) :
    IVFBase(res,
            metric,
            metricArg,
            quantizer,
            indicesOptions,
            space),
    useResidual_(useResidual),
    scalarQ_(scalarQ ? new GpuScalarQuantizer(res, *scalarQ) : nullptr) {
}

IVFFlat::~IVFFlat() {
}

size_t
IVFFlat::getGpuVectorsEncodingSize_(int numVecs) const {
  return (size_t) numVecs *
    // scalar size per dimension
    (scalarQ_ ? scalarQ_->code_size : sizeof(float)) *
    // number of dimensions
    getDim();
}

size_t
IVFFlat::getCpuVectorsEncodingSize_(int numVecs) const {
  return (size_t) numVecs *
    // scalar size per dimension
    (scalarQ_ ? scalarQ_->code_size : sizeof(float)) *
    // number of dimensions
    getDim();
}

std::vector<unsigned char>
IVFFlat::translateCodesToGpu_(std::vector<unsigned char> codes,
                              size_t numVecs) const {
  // nothing to do
  return codes;
}

std::vector<unsigned char>
IVFFlat::translateCodesFromGpu_(std::vector<unsigned char> codes,
                                size_t numVecs) const {
  // nothing to do
  return codes;
}

void
IVFFlat::appendVectors_(Tensor<float, 2, true>& vecs,
                        Tensor<long, 1, true>& indices,
                        Tensor<int, 1, true>& listIds,
                        Tensor<int, 1, true>& listOffset,
                        cudaStream_t stream) {
  //
  // Append the new encodings
  //

  // Calculate residuals for these vectors, if needed
  DeviceTensor<float, 2, true> residuals(
    resources_, makeTempAlloc(AllocType::Other, stream), {vecs.getSize(0), dim_});

  if (useResidual_) {
    quantizer_->computeResidual(vecs, listIds, residuals);
  }

  // Now, for each list to which a vector is being assigned, write it
  runIVFFlatInvertedListAppend(listIds,
                               listOffset,
                               vecs,
                               indices,
                               useResidual_,
                               residuals,
                               scalarQ_.get(),
                               deviceListDataPointers_,
                               deviceListIndexPointers_,
                               indicesOptions_,
                               stream);
}

void
IVFFlat::query(Tensor<float, 2, true>& queries,
               int nprobe,
               int k,
               Tensor<float, 2, true>& outDistances,
               Tensor<long, 2, true>& outIndices) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // These are caught at a higher level
  FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
  FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);

  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

  // Reserve space for the quantized information
  DeviceTensor<float, 2, true> coarseDistances(
    resources_, makeTempAlloc(AllocType::Other, stream), {queries.getSize(0), nprobe});
  DeviceTensor<int, 2, true> coarseIndices(
    resources_, makeTempAlloc(AllocType::Other, stream), {queries.getSize(0), nprobe});

  // Find the `nprobe` closest lists; we can use int indices both
  // internally and externally
  quantizer_->query(queries,
                    nprobe,
                    metric_,
                    metricArg_,
                    coarseDistances,
                    coarseIndices,
                    false);

  DeviceTensor<float, 3, true> residualBase(
    resources_, makeTempAlloc(AllocType::Other, stream),
    {queries.getSize(0), nprobe, dim_});

  if (useResidual_) {
    // Reconstruct vectors from the quantizer
    quantizer_->reconstruct(coarseIndices, residualBase);
  }

  runIVFFlatScan(queries,
                 coarseIndices,
                 deviceListDataPointers_,
                 deviceListIndexPointers_,
                 indicesOptions_,
                 deviceListLengths_,
                 maxListLength_,
                 k,
                 metric_,
                 useResidual_,
                 residualBase,
                 scalarQ_.get(),
                 outDistances,
                 outIndices,
                 resources_);

  // If the GPU isn't storing indices (they are on the CPU side), we
  // need to perform the re-mapping here
  // FIXME: we might ultimately be calling this function with inputs
  // from the CPU, these are unnecessary copies
  if (indicesOptions_ == INDICES_CPU) {
    HostTensor<long, 2, true> hostOutIndices(outIndices, stream);

    ivfOffsetToUserIndex(hostOutIndices.data(),
                         numLists_,
                         hostOutIndices.getSize(0),
                         hostOutIndices.getSize(1),
                         listOffsetToUserIndex_);

    // Copy back to GPU, since the input to this function is on the
    // GPU
    outIndices.copyFrom(hostOutIndices, stream);
  }
}

} } // namespace
