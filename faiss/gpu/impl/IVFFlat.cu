/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
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

IVFFlat::IVFFlat(
        GpuResources* res,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFBase(res,
                  dim,
                  nlist,
                  metric,
                  metricArg,
                  useResidual,
                  interleavedLayout,
                  indicesOptions,
                  space),
          scalarQ_(scalarQ ? new GpuScalarQuantizer(res, *scalarQ) : nullptr) {}

IVFFlat::~IVFFlat() {}

size_t IVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    if (interleavedLayout_) {
        // bits per scalar code
        idx_t bits = scalarQ_ ? scalarQ_->bits : 32 /* float */;

        int warpSize = getWarpSizeCurrentDevice();

        // bytes to encode a block of warpSize vectors (single dimension)
        idx_t bytesPerDimBlock = bits * warpSize / 8;

        // bytes to fully encode warpSize vectors
        idx_t bytesPerBlock = bytesPerDimBlock * dim_;

        // number of blocks of warpSize vectors we have
        idx_t numBlocks = utils::divUp(numVecs, warpSize);

        // total size to encode numVecs
        return bytesPerBlock * numBlocks;
    } else {
        size_t sizePerVector =
                (scalarQ_ ? scalarQ_->code_size : sizeof(float) * dim_);

        return (size_t)numVecs * sizePerVector;
    }
}

size_t IVFFlat::getCpuVectorsEncodingSize_(idx_t numVecs) const {
    size_t sizePerVector =
            (scalarQ_ ? scalarQ_->code_size : sizeof(float) * dim_);

    return (size_t)numVecs * sizePerVector;
}

std::vector<uint8_t> IVFFlat::translateCodesToGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    if (!interleavedLayout_) {
        // same format
        return codes;
    }

    int bitsPerCode = scalarQ_ ? scalarQ_->bits : 32;

    auto up =
            unpackNonInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

std::vector<uint8_t> IVFFlat::translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        idx_t numVecs) const {
    if (!interleavedLayout_) {
        // same format
        return codes;
    }

    int bitsPerCode = scalarQ_ ? scalarQ_->bits : 32;

    auto up = unpackInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
    return packNonInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

void IVFFlat::appendVectors_(
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfCentroidResiduals,
        Tensor<idx_t, 1, true>& indices,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<idx_t, 1, true>& vectorsByUniqueList,
        Tensor<idx_t, 1, true>& uniqueListVectorStart,
        Tensor<idx_t, 1, true>& uniqueListStartOffset,
        Tensor<idx_t, 1, true>& listIds,
        Tensor<idx_t, 1, true>& listOffset,
        cudaStream_t stream) {
    //
    // Append the new encodings
    //

    // Append indices to the IVF lists
    runIVFIndicesAppend(
            listIds,
            listOffset,
            indices,
            indicesOptions_,
            deviceListIndexPointers_,
            stream);

    // Append the encoded vectors to the IVF lists
    if (interleavedLayout_) {
        runIVFFlatInterleavedAppend(
                listIds,
                listOffset,
                uniqueLists,
                vectorsByUniqueList,
                uniqueListVectorStart,
                uniqueListStartOffset,
                useResidual_ ? ivfCentroidResiduals : vecs,
                scalarQ_.get(),
                deviceListDataPointers_,
                resources_,
                stream);
    } else {
        runIVFFlatAppend(
                listIds,
                listOffset,
                useResidual_ ? ivfCentroidResiduals : vecs,
                scalarQ_.get(),
                deviceListDataPointers_,
                stream);
    }
}

void IVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // These are caught at a higher level
    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
    nprobe = int(std::min(idx_t(nprobe), getNumLists()));

    FAISS_ASSERT(queries.getSize(1) == dim_);

    FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
    FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

    // Reserve space for the quantized information
    DeviceTensor<float, 2, true> coarseDistances(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});
    DeviceTensor<idx_t, 2, true> coarseIndices(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});
    // in case we also want/need residuals, we need the original centroids as
    // well
    // FIXME: why centroids instead of calculating residuals in one go?
    DeviceTensor<float, 3, true> residualBase(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe, dim_});

    searchCoarseQuantizer_(
            coarseQuantizer,
            nprobe,
            queries,
            coarseDistances,
            coarseIndices,
            nullptr,
            // we need the IVF centroids to which vectors were assigned if
            // vectors are encoded using the residual
            useResidual_ ? &residualBase : nullptr);

    searchImpl_(
            queries,
            coarseDistances,
            coarseIndices,
            residualBase,
            k,
            outDistances,
            outIndices,
            false);
}

void IVFFlat::searchPreassigned(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfDistances,
        Tensor<idx_t, 2, true>& ivfAssignments,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool storePairs) {
    FAISS_ASSERT(ivfDistances.getSize(0) == vecs.getSize(0));
    FAISS_ASSERT(ivfAssignments.getSize(0) == vecs.getSize(0));
    FAISS_ASSERT(outDistances.getSize(0) == vecs.getSize(0));
    FAISS_ASSERT(outIndices.getSize(0) == vecs.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == dim_);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto nprobe = ivfAssignments.getSize(1);
    FAISS_ASSERT(nprobe <= numLists_);

    // Based on the IVF assignments, we need the IVF centroids to which vectors
    // were assigned
    // FIXME: IVFPQ doesn't need this information as it has direct reference to
    // all IVF centroids and within the various kernels can look it up by index
    // as needed. Can we convert IVFFlat to do the same thing?
    DeviceTensor<float, 3, true> ivfCentroids(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), nprobe, dim_});

    auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
    if (gpuQuantizer) {
        // We can pass device pointers directly
        gpuQuantizer->reconstruct_batch(
                vecs.getSize(0) * nprobe,
                ivfAssignments.data(),
                ivfCentroids.data());
    } else {
        // CPU coarse quantizer
        auto cpuIVFCentroids =
                std::vector<float>(vecs.getSize(0) * nprobe * dim_);

        // We need to copy `ivfAssignments` to the CPU, in order to pass to a
        // CPU index
        auto cpuIVFAssignments = ivfAssignments.copyToVector(stream);

        coarseQuantizer->reconstruct_batch(
                vecs.getSize(0) * nprobe,
                cpuIVFAssignments.data(),
                cpuIVFCentroids.data());

        ivfCentroids.copyFrom(cpuIVFCentroids, stream);
    }

    searchImpl_(
            vecs,
            ivfDistances,
            ivfAssignments,
            ivfCentroids,
            k,
            outDistances,
            outIndices,
            storePairs);
}

void IVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* out) {
    if (ni == 0) {
        // nothing to do
        return;
    }

    int warpSize = getWarpSizeCurrentDevice();
    auto stream = resources_->getDefaultStreamCurrentDevice();

    for (idx_t list_no = 0; list_no < numLists_; list_no++) {
        size_t list_size = deviceListData_[list_no]->numVecs;

        auto idlist = getListIndices(list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            // vector data in the non-interleaved format is laid out like:
            // v0d0 v0d1 ... v0d(dim-1) v1d0 v1d1 ... v1d(dim-1)

            // vector data in the interleaved format is laid out like:
            // (v0d0 v1d0 ... v31d0) (v0d1 v1d1 ... v31d1)
            // (v0d(dim - 1) ... v31d(dim-1))
            // (v32d0 v33d0 ... v63d0) (... v63d(dim-1)) (v64d0 ...)

            // where vectors are chunked into groups of 32, and each dimension
            // for each of the 32 vectors is contiguous

            auto vectorChunk = offset / warpSize;
            auto vectorWithinChunk = offset % warpSize;

            auto listDataPtr = (float*)deviceListData_[list_no]->data.data();
            listDataPtr += vectorChunk * warpSize * dim_ + vectorWithinChunk;

            for (int d = 0; d < dim_; ++d) {
                fromDevice<float>(
                        listDataPtr + warpSize * d,
                        out + (id - i0) * dim_ + d,
                        1,
                        stream);
            }
        }
    }
}

void IVFFlat::searchImpl_(
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        Tensor<float, 3, true>& ivfCentroids,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool storePairs) {
    FAISS_ASSERT(storePairs == false);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (interleavedLayout_) {
        runIVFInterleavedScan(
                queries,
                coarseIndices,
                deviceListDataPointers_,
                deviceListIndexPointers_,
                indicesOptions_,
                deviceListLengths_,
                k,
                metric_,
                useResidual_,
                ivfCentroids,
                scalarQ_.get(),
                outDistances,
                outIndices,
                resources_);
    } else {
        runIVFFlatScan(
                queries,
                coarseIndices,
                deviceListDataPointers_,
                deviceListIndexPointers_,
                indicesOptions_,
                deviceListLengths_,
                maxListLength_,
                k,
                metric_,
                useResidual_,
                ivfCentroids,
                scalarQ_.get(),
                outDistances,
                outIndices,
                resources_);
    }

    // If the GPU isn't storing indices (they are on the CPU side), we
    // need to perform the re-mapping here
    // FIXME: we might ultimately be calling this function with inputs
    // from the CPU, these are unnecessary copies
    if (indicesOptions_ == INDICES_CPU) {
        HostTensor<idx_t, 2, true> hostOutIndices(outIndices, stream);

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
