/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFPQ.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/PQCodeDistances.cuh>
#include <faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh>
#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/NoTypeTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <type_traits>
#include <unordered_map>

namespace faiss {
namespace gpu {

IVFPQ::IVFPQ(
        GpuResources* resources,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useFloat16LookupTables,
        bool useMMCodeDistance,
        bool interleavedLayout,
        float* pqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFBase(resources,
                  dim,
                  nlist,
                  metric,
                  metricArg,
                  // we use IVF cell residuals for encoding vectors
                  true,
                  interleavedLayout,
                  indicesOptions,
                  space),
          numSubQuantizers_(numSubQuantizers),
          bitsPerSubQuantizer_(bitsPerSubQuantizer),
          numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
          dimPerSubQuantizer_(dim_ / numSubQuantizers),
          useFloat16LookupTables_(useFloat16LookupTables),
          useMMCodeDistance_(useMMCodeDistance),
          precomputedCodes_(false) {
    FAISS_ASSERT(pqCentroidData);

    FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
    FAISS_ASSERT(dim_ % numSubQuantizers_ == 0);
    FAISS_ASSERT(
            interleavedLayout || isSupportedPQCodeLength(numSubQuantizers_));

    setPQCentroids_(pqCentroidData);
}

IVFPQ::~IVFPQ() {}

bool IVFPQ::isSupportedPQCodeLength(int size) {
    switch (size) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 8:
        case 12:
        case 16:
        case 20:
        case 24:
        case 28:
        case 32:
        case 40:
        case 48:
        case 56: // only supported with float16
        case 64: // only supported with float16
        case 96: // only supported with float16
            return true;
        default:
            return false;
    }
}

void IVFPQ::setPrecomputedCodes(Index* quantizer, bool enable) {
    if (enable && metric_ == MetricType::METRIC_INNER_PRODUCT) {
        fprintf(stderr,
                "Precomputed codes are not needed for GpuIndexIVFPQ "
                "with METRIC_INNER_PRODUCT");
        return;
    }

    if (precomputedCodes_ != enable) {
        precomputedCodes_ = enable;

        if (precomputedCodes_) {
            precomputeCodes_(quantizer);
        } else {
            // Clear out old precomputed code data
            precomputedCode_ = DeviceTensor<float, 3, true>();
            precomputedCodeHalf_ = DeviceTensor<half, 3, true>();
        }
    }
}

Tensor<float, 3, true> IVFPQ::getPQCentroids() {
    return pqCentroidsMiddleCode_;
}

void IVFPQ::appendVectors_(
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfCentroidResiduals,
        Tensor<idx_t, 1, true>& indices,
        Tensor<idx_t, 1, true>& uniqueLists,
        Tensor<int, 1, true>& vectorsByUniqueList,
        Tensor<int, 1, true>& uniqueListVectorStart,
        Tensor<int, 1, true>& uniqueListStartOffset,
        Tensor<idx_t, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        cudaStream_t stream) {
    //
    // Determine the encodings of the vectors
    //

    // For now we are restricted to <= 8 bits per code (hence uint8_t in the
    // encodings)
    FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);

    DeviceTensor<uint8_t, 2, true> encodings(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), numSubQuantizers_});

    {
        // Residuals are in the form
        // (vec x numSubQuantizer x dimPerSubQuantizer)
        // transpose to
        // (numSubQuantizer x vec x dimPerSubQuantizer)
        auto residualsView = ivfCentroidResiduals.view<3>(
                {ivfCentroidResiduals.getSize(0),
                 numSubQuantizers_,
                 dimPerSubQuantizer_});

        DeviceTensor<float, 3, true> residualsTranspose(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_,
                 ivfCentroidResiduals.getSize(0),
                 dimPerSubQuantizer_});

        runTransposeAny(residualsView, 0, 1, residualsTranspose, stream);

        // Get the product quantizer centroids in the form
        // (numSubQuantizer x numSubQuantizerCodes x dimPerSubQuantizer)
        // which is pqCentroidsMiddleCode_

        // We now have a batch operation to find the top-1 distances:
        // batch size: numSubQuantizer
        // centroids: (numSubQuantizerCodes x dimPerSubQuantizer)
        // residuals: (vec x dimPerSubQuantizer)
        // => (numSubQuantizer x vec x 1)
        DeviceTensor<float, 3, true> closestSubQDistance(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, ivfCentroidResiduals.getSize(0), 1});
        DeviceTensor<int, 3, true> closestSubQIndex(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, ivfCentroidResiduals.getSize(0), 1});

        for (int subQ = 0; subQ < numSubQuantizers_; ++subQ) {
            auto closestSubQDistanceView = closestSubQDistance[subQ].view();
            auto closestSubQIndexView = closestSubQIndex[subQ].view();

            auto pqCentroidsMiddleCodeView =
                    pqCentroidsMiddleCode_[subQ].view();
            auto residualsTransposeView = residualsTranspose[subQ].view();

            runL2Distance(
                    resources_,
                    stream,
                    pqCentroidsMiddleCodeView,
                    true,    // pqCentroidsMiddleCodeView is row major
                    nullptr, // no precomputed norms
                    residualsTransposeView,
                    true, // residualsTransposeView is row major
                    1,
                    closestSubQDistanceView,
                    closestSubQIndexView,
                    // We don't care about distances
                    true);
        }

        // The L2 distance function only returns int32 indices. As we are
        // restricted to <= 8 bits per code, convert to uint8
        auto closestSubQIndex8 = convertTensorTemporary<int, uint8_t, 3>(
                resources_, stream, closestSubQIndex);

        // Now, we have the nearest sub-q centroid for each slice of the
        // residual vector.
        auto closestSubQIndex8View = closestSubQIndex8.view<2>(
                {numSubQuantizers_, ivfCentroidResiduals.getSize(0)});

        // The encodings are finally a transpose of this data
        runTransposeAny(closestSubQIndex8View, 0, 1, encodings, stream);
    }

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
        runIVFPQInterleavedAppend(
                listIds,
                listOffset,
                uniqueLists,
                vectorsByUniqueList,
                uniqueListVectorStart,
                uniqueListStartOffset,
                bitsPerSubQuantizer_,
                encodings,
                deviceListDataPointers_,
                stream);
    } else {
        runIVFPQAppend(
                listIds,
                listOffset,
                encodings,
                deviceListDataPointers_,
                stream);
    }
}

size_t IVFPQ::getGpuVectorsEncodingSize_(int numVecs) const {
    if (interleavedLayout_) {
        // bits per PQ code
        int bits = bitsPerSubQuantizer_;

        // bytes to encode a block of 32 vectors (single PQ code)
        int bytesPerDimBlock = bits * 32 / 8;

        // bytes to fully encode 32 vectors
        int bytesPerBlock = bytesPerDimBlock * numSubQuantizers_;

        // number of blocks of 32 vectors we have
        int numBlocks = utils::divUp(numVecs, 32);

        // total size to encode numVecs
        return bytesPerBlock * numBlocks;
    } else {
        return (size_t)numVecs * numSubQuantizers_;
    }
}

size_t IVFPQ::getCpuVectorsEncodingSize_(int numVecs) const {
    size_t sizePerVector =
            utils::divUp(numSubQuantizers_ * bitsPerSubQuantizer_, 8);

    return (size_t)numVecs * sizePerVector;
}

// Convert the CPU layout to the GPU layout
std::vector<uint8_t> IVFPQ::translateCodesToGpu_(
        std::vector<uint8_t> codes,
        size_t numVecs) const {
    if (!interleavedLayout_) {
        return codes;
    }

    auto up = unpackNonInterleaved(
            std::move(codes), numVecs, numSubQuantizers_, bitsPerSubQuantizer_);
    return packInterleaved(
            std::move(up), numVecs, numSubQuantizers_, bitsPerSubQuantizer_);
}

// Conver the GPU layout to the CPU layout
std::vector<uint8_t> IVFPQ::translateCodesFromGpu_(
        std::vector<uint8_t> codes,
        size_t numVecs) const {
    if (!interleavedLayout_) {
        return codes;
    }

    auto up = unpackInterleaved(
            std::move(codes), numVecs, numSubQuantizers_, bitsPerSubQuantizer_);
    return packNonInterleaved(
            std::move(up), numVecs, numSubQuantizers_, bitsPerSubQuantizer_);
}

void IVFPQ::setPQCentroids_(float* data) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    size_t pqSize =
            numSubQuantizers_ * numSubQuantizerCodes_ * dimPerSubQuantizer_;

    // Make sure the data is on the host
    // FIXME: why are we doing this?
    thrust::host_vector<float> hostMemory;
    hostMemory.insert(hostMemory.end(), data, data + pqSize);

    HostTensor<float, 3, true> pqHost(
            hostMemory.data(),
            {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});

    DeviceTensor<float, 3, true> pqDeviceTranspose(
            resources_,
            makeDevAlloc(AllocType::Quantizer, stream),
            {numSubQuantizers_, dimPerSubQuantizer_, numSubQuantizerCodes_});

    {
        // Only needed for the duration of the transposition
        DeviceTensor<float, 3, true> pqDevice(
                resources_,
                makeTempAlloc(AllocType::Quantizer, stream),
                pqHost);

        runTransposeAny(pqDevice, 1, 2, pqDeviceTranspose, stream);
    }

    pqCentroidsInnermostCode_ = std::move(pqDeviceTranspose);

    // Also maintain the PQ centroids in the form
    // (sub q)(code id)(sub dim)
    DeviceTensor<float, 3, true> pqCentroidsMiddleCode(
            resources_,
            makeDevAlloc(AllocType::Quantizer, stream),
            {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});

    runTransposeAny(
            pqCentroidsInnermostCode_, 1, 2, pqCentroidsMiddleCode, stream);

    pqCentroidsMiddleCode_ = std::move(pqCentroidsMiddleCode);
}

void IVFPQ::precomputeCodes_(Index* quantizer) {
    FAISS_ASSERT(metric_ == MetricType::METRIC_L2);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    //
    //    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
    //        ---------------   ---------------------------       -------
    //            term 1                 term 2                   term 3
    //

    // Terms 1 and 3 are available only at query time. We compute term 2
    // here.

    // Compute 2 * (y_C|y_R) via batch matrix multiplication
    // batch size (sub q) x {(centroid id)(sub dim) x (code id)(sub dim)'}
    //         => (sub q) x {(centroid id)(code id)}
    //         => (sub q)(centroid id)(code id)

    // Whether or not there is a CPU or GPU coarse quantizer, updateQuantizer()
    // should have been called to reconstruct as float32 the IVF centroids to
    // have the data available on the GPU
    FAISS_THROW_IF_NOT_MSG(
            ivfCentroids_.getSize(0) == getNumLists() &&
                    ivfCentroids_.getSize(1) == getDim(),
            "IVFPQ::precomputeCodes: coarse quantizer data "
            "not synchronized on GPU; must call updateQuantizer() "
            "before continuing");

    // View (centroid id)(dim) as
    //      (centroid id)(sub q)(dim)
    // Transpose (centroid id)(sub q)(sub dim) to
    //           (sub q)(centroid id)(sub dim)

    // Create the coarse PQ product
    DeviceTensor<float, 3, true> coarsePQProduct(
            resources_,
            makeTempAlloc(AllocType::QuantizerPrecomputedCodes, stream),
            {numSubQuantizers_,
             ivfCentroids_.getSize(0),
             numSubQuantizerCodes_});

    {
        auto centroidView = ivfCentroids_.template view<3>(
                {ivfCentroids_.getSize(0),
                 numSubQuantizers_,
                 dimPerSubQuantizer_});

        // This is only needed temporarily
        DeviceTensor<float, 3, true> centroidsTransposed(
                resources_,
                makeTempAlloc(AllocType::QuantizerPrecomputedCodes, stream),
                {numSubQuantizers_,
                 ivfCentroids_.getSize(0),
                 dimPerSubQuantizer_});

        runTransposeAny(centroidView, 0, 1, centroidsTransposed, stream);

        runBatchMatrixMult(
                coarsePQProduct,
                false,
                centroidsTransposed,
                false,
                pqCentroidsMiddleCode_,
                true,
                2.0f,
                0.0f,
                resources_->getBlasHandleCurrentDevice(),
                stream);
    }

    // Transpose (sub q)(centroid id)(code id) to
    //           (centroid id)(sub q)(code id)
    // This will become our precomputed code output
    DeviceTensor<float, 3, true> coarsePQProductTransposed(
            resources_,
            makeDevAlloc(AllocType::QuantizerPrecomputedCodes, stream),
            {ivfCentroids_.getSize(0),
             numSubQuantizers_,
             numSubQuantizerCodes_});
    runTransposeAny(coarsePQProduct, 0, 1, coarsePQProductTransposed, stream);

    // View (centroid id)(sub q)(code id) as
    //      (centroid id)(sub q * code id)
    auto coarsePQProductTransposedView = coarsePQProductTransposed.view<2>(
            {ivfCentroids_.getSize(0),
             numSubQuantizers_ * numSubQuantizerCodes_});

    // Sum || y_R ||^2 + 2 * (y_C|y_R)
    // i.e., add norms                              (sub q * code id)
    // along columns of inner product  (centroid id)(sub q * code id)
    {
        // Compute ||y_R||^2 by treating
        // (sub q)(code id)(sub dim) as (sub q * code id)(sub dim)
        auto pqCentroidsMiddleCodeView = pqCentroidsMiddleCode_.view<2>(
                {numSubQuantizers_ * numSubQuantizerCodes_,
                 dimPerSubQuantizer_});
        DeviceTensor<float, 1, true> subQuantizerNorms(
                resources_,
                makeTempAlloc(AllocType::QuantizerPrecomputedCodes, stream),
                {numSubQuantizers_ * numSubQuantizerCodes_});

        runL2Norm(
                pqCentroidsMiddleCodeView,
                true,
                subQuantizerNorms,
                true,
                stream);

        runSumAlongColumns(
                subQuantizerNorms, coarsePQProductTransposedView, stream);
    }

    // We added into the view, so `coarsePQProductTransposed` is now our
    // precomputed term 2.
    if (useFloat16LookupTables_) {
        precomputedCodeHalf_ = DeviceTensor<half, 3, true>(
                resources_,
                makeDevAlloc(AllocType::QuantizerPrecomputedCodes, stream),
                {ivfCentroids_.getSize(0),
                 numSubQuantizers_,
                 numSubQuantizerCodes_});

        convertTensor(stream, coarsePQProductTransposed, precomputedCodeHalf_);
    } else {
        precomputedCode_ = std::move(coarsePQProductTransposed);
    }
}

void IVFPQ::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    // These are caught at a higher level
    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    nprobe = std::min(nprobe, (int)getNumLists());

    FAISS_ASSERT(queries.getSize(1) == dim_);
    FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
    FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

    // Reserve space for the closest coarse centroids
    DeviceTensor<float, 2, true> coarseDistances(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});
    DeviceTensor<idx_t, 2, true> coarseIndices(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});

    searchCoarseQuantizer_(
            coarseQuantizer,
            nprobe,
            queries,
            coarseDistances,
            coarseIndices,
            nullptr /* don't need IVF centroid residuals */,
            nullptr /* don't need IVF centroids */);

    searchImpl_(
            queries,
            coarseDistances,
            coarseIndices,
            k,
            outDistances,
            outIndices,
            false);
}

void IVFPQ::searchPreassigned(
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

    searchImpl_(
            vecs,
            ivfDistances,
            ivfAssignments,
            k,
            outDistances,
            outIndices,
            storePairs);
}

void IVFPQ::searchImpl_(
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool storePairs) {
    FAISS_ASSERT(storePairs == false);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (precomputedCodes_) {
        FAISS_ASSERT(metric_ == MetricType::METRIC_L2);

        runPQPrecomputedCodes_(
                queries,
                coarseDistances,
                coarseIndices,
                k,
                outDistances,
                outIndices);
    } else {
        runPQNoPrecomputedCodes_(
                queries,
                coarseDistances,
                coarseIndices,
                k,
                outDistances,
                outIndices);
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

void IVFPQ::runPQPrecomputedCodes_(
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    FAISS_ASSERT(metric_ == MetricType::METRIC_L2);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    // Compute precomputed code term 3, - 2 * (x|y_R)
    // This is done via batch MM
    // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
    // {sub q} x {(query id)(code id)}
    DeviceTensor<float, 3, true> term3Transposed(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), numSubQuantizers_, numSubQuantizerCodes_});

    // These allocations within are only temporary, so release them when
    // we're done to maximize free space
    {
        auto querySubQuantizerView = queries.view<3>(
                {queries.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});
        DeviceTensor<float, 3, true> queriesTransposed(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, queries.getSize(0), dimPerSubQuantizer_});
        runTransposeAny(querySubQuantizerView, 0, 1, queriesTransposed, stream);

        DeviceTensor<float, 3, true> term3(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, queries.getSize(0), numSubQuantizerCodes_});

        runBatchMatrixMult(
                term3,
                false,
                queriesTransposed,
                false,
                pqCentroidsMiddleCode_,
                true,
                -2.0f,
                0.0f,
                resources_->getBlasHandleCurrentDevice(),
                stream);

        runTransposeAny(term3, 0, 1, term3Transposed, stream);
    }

    NoTypeTensor<3, true> term2;
    NoTypeTensor<3, true> term3;
    DeviceTensor<half, 3, true> term3Half;

    if (useFloat16LookupTables_) {
        term3Half = convertTensorTemporary<float, half, 3>(
                resources_, stream, term3Transposed);

        term2 = NoTypeTensor<3, true>(precomputedCodeHalf_);
        term3 = NoTypeTensor<3, true>(term3Half);
    } else {
        term2 = NoTypeTensor<3, true>(precomputedCode_);
        term3 = NoTypeTensor<3, true>(term3Transposed);
    }

    runPQScanMultiPassPrecomputed(
            queries,
            coarseDistances, // term 1
            term2,           // term 2
            term3,           // term 3
            coarseIndices,
            useFloat16LookupTables_,
            interleavedLayout_,
            bitsPerSubQuantizer_,
            numSubQuantizers_,
            numSubQuantizerCodes_,
            deviceListDataPointers_,
            deviceListIndexPointers_,
            indicesOptions_,
            deviceListLengths_,
            maxListLength_,
            k,
            outDistances,
            outIndices,
            resources_);
}

void IVFPQ::runPQNoPrecomputedCodes_(
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    runPQScanMultiPassNoPrecomputed(
            queries,
            ivfCentroids_,
            pqCentroidsInnermostCode_,
            coarseDistances,
            coarseIndices,
            useFloat16LookupTables_,
            useMMCodeDistance_,
            interleavedLayout_,
            bitsPerSubQuantizer_,
            numSubQuantizers_,
            numSubQuantizerCodes_,
            deviceListDataPointers_,
            deviceListIndexPointers_,
            indicesOptions_,
            deviceListLengths_,
            maxListLength_,
            k,
            metric_,
            outDistances,
            outIndices,
            resources_);
}

} // namespace gpu
} // namespace faiss
