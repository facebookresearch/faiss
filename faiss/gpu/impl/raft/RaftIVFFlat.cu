/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <raft/core/handle.hpp>
#include <raft/spatial/knn/ivf_flat.cuh>

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/raft/RaftIVFFlat.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>

namespace faiss {
namespace gpu {

RaftIVFFlat::RaftIVFFlat(
        GpuResources* res,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFFlat(res,
                  dim,
                  nlist,
                  metric,
                  metricArg,
                  useResidual,
                  scalarQ,
                  interleavedLayout,
                  indicesOptions,
                  space){}

RaftIVFFlat::~RaftIVFFlat() {}

size_t RaftIVFFlat::getGpuVectorsEncodingSize_(int numVecs) const {
    if (interleavedLayout_) {
        // bits per scalar code
        int bits = scalarQ_ ? scalarQ_->bits : 32 /* float */;

        // bytes to encode a block of 32 vectors (single dimension)
        int bytesPerDimBlock = bits * 32 / 8;

        // bytes to fully encode 32 vectors
        int bytesPerBlock = bytesPerDimBlock * dim_;

        // number of blocks of 32 vectors we have
        int numBlocks = utils::divUp(numVecs, 32);

        // total size to encode numVecs
        return bytesPerBlock * numBlocks;
    } else {
        size_t sizePerVector =
                (scalarQ_ ? scalarQ_->code_size : sizeof(float) * dim_);

        return (size_t)numVecs * sizePerVector;
    }
}

size_t RaftIVFFlat::getCpuVectorsEncodingSize_(int numVecs) const {
    size_t sizePerVector =
            (scalarQ_ ? scalarQ_->code_size : sizeof(float) * dim_);

    return (size_t)numVecs * sizePerVector;
}

std::vector<uint8_t> RaftIVFFlat::translateCodesToGpu_(
        std::vector<uint8_t> codes,
        size_t numVecs) const {
    if (!interleavedLayout_) {
        // same format
        return codes;
    }

    int bitsPerCode = scalarQ_ ? scalarQ_->bits : 32;

    auto up =
            unpackNonInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

std::vector<uint8_t> RaftIVFFlat::translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        size_t numVecs) const {
    if (!interleavedLayout_) {
        // same format
        return codes;
    }

    int bitsPerCode = scalarQ_ ? scalarQ_->bits : 32;

    auto up = unpackInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packNonInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}


void RaftIVFFlat::appendVectors_(
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfCentroidResiduals,
        Tensor<Index::idx_t, 1, true>& indices,
        Tensor<Index::idx_t, 1, true>& uniqueLists,
        Tensor<int, 1, true>& vectorsByUniqueList,
        Tensor<int, 1, true>& uniqueListVectorStart,
        Tensor<int, 1, true>& uniqueListStartOffset,
        Tensor<Index::idx_t, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        cudaStream_t stream) {
    //
    // Append the new encodings
    //

    // TODO: Fill in this logic here
}

void RaftIVFFlat::searchImpl_(
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<Index::idx_t, 2, true>& coarseIndices,
        Tensor<float, 3, true>& ivfCentroids,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices,
        bool storePairs) {
    FAISS_ASSERT(storePairs == false);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    // TODO: Fill in this logic here.

//    // Device is already set in GpuIndex::search
//    FAISS_ASSERT(raft_knn_index.has_value());
//    FAISS_ASSERT(n > 0);
//    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);
//
//    raft::spatial::knn::ivf_flat::search_params pams;
//    pams.n_probes = nprobe;
//    raft::spatial::knn::ivf_flat::search<float, faiss::Index::idx_t>(
//            raft_handle,
//            pams,
//            *raft_knn_index,
//            const_cast<float*>(x),
//            static_cast<std::uint32_t>(n),
//            static_cast<std::uint32_t>(k),
//            labels,
//            distances);
//
//    raft_handle.sync_stream();


    // If the GPU isn't storing indices (they are on the CPU side), we
    // need to perform the re-mapping here
    // FIXME: we might ultimately be calling this function with inputs
    // from the CPU, these are unnecessary copies
    if (indicesOptions_ == INDICES_CPU) {
        HostTensor<Index::idx_t, 2, true> hostOutIndices(outIndices, stream);

        ivfOffsetToUserIndex(
                hostOutIndices.data(),
                numLists_,
                hostOutIndices.getSize(0),
                hostOutIndices.getSize(1),
                listOffsetToUserIndex_);

        // Copy back to GPU, since the input to this function is on the
        // GPU
        outIndices.copyFrom(hostOutIndices, stream);
    }
}

} // namespace gpu
} // namespace faiss
