/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/IMIAppend.cuh>
#include <faiss/gpu/impl/IMIPQ.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
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
#include <iostream>
#include <limits>
#include <unordered_map>

namespace faiss {
namespace gpu {

IMIPQ::IMIPQ(
        GpuResources* resources,
        MultiIndex2* quantizer,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useMMCodeDistance,
        bool interleavedLayout,
        bool precomputeCodesOnCpu,
        float* pqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IMIBase(resources,
                  quantizer,
                  interleavedLayout,
                  indicesOptions,
                  space),
          numSubQuantizers_(numSubQuantizers),
          bitsPerSubQuantizer_(bitsPerSubQuantizer),
          numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
          dimPerSubQuantizer_(dim_ / numSubQuantizers),
          useFloat16LookupTables_(false),
          useMMCodeDistance_(useMMCodeDistance),
          precomputedCodes_(false),
          precomputeCodesOnCpu_(precomputeCodesOnCpu_) {
    FAISS_ASSERT(pqCentroidData);

    FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
    FAISS_ASSERT(numSubQuantizerCodes_ <= 256);
    FAISS_ASSERT(dim_ % quantizer_->getNumCodebooks() == 0);
    FAISS_ASSERT(numSubQuantizers_ % quantizer_->getNumCodebooks() == 0);
    FAISS_ASSERT(dim_ % numSubQuantizers_ == 0);
    FAISS_ASSERT(dimPerSubQuantizer_ > 0);
    FAISS_ASSERT(
            interleavedLayout || isSupportedPQCodeLength(numSubQuantizers_));
    FAISS_ASSERT(!interleavedLayout); // not supported yet

    setPQCentroids_(pqCentroidData);
    setPrecomputedCodes(true);
}

IMIPQ::~IMIPQ() {}

bool IMIPQ::isSupportedPQCodeLength(int size) {
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

size_t IMIPQ::calcVectorsEncodingMemorySpaceSize(
        int numVecs,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool interleavedLayout) {
    if (interleavedLayout) {
        // bits per PQ code
        int bits = bitsPerSubQuantizer;

        // bytes to encode a block of 32 vectors (single PQ code)
        int bytesPerDimBlock = bits * 32 / 8;

        // bytes to fully encode 32 vectors
        int bytesPerBlock = bytesPerDimBlock * numSubQuantizers;

        // number of blocks of 32 vectors we have
        int numBlocks = utils::divUp(numVecs, 32);

        // total size to encode numVecs
        return bytesPerBlock * numBlocks;
    } else {
        return (size_t)numVecs * numSubQuantizers;
    }
}

size_t IMIPQ::calcIndicesMemorySpaceSize(int numVecs, IndicesOptions options) {
    if ((options == INDICES_32_BIT) || (options == INDICES_64_BIT)) {
        return numVecs *
                (options == INDICES_32_BIT ? sizeof(int) : sizeof(idx_t));
    }

    return 0;
}

size_t IMIPQ::calcMemorySpaceSize(
        int numVecs,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool interleavedLayout,
        IndicesOptions options) {
    return calcVectorsEncodingMemorySpaceSize(
                   numVecs,
                   numSubQuantizers,
                   bitsPerSubQuantizer,
                   interleavedLayout) +
            calcIndicesMemorySpaceSize(numVecs, options);
}

std::unordered_map<AllocType, size_t> IMIPQ::getAllocSizePerTypeInfo(
        int numVecs,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool interleavedLayout,
        IndicesOptions options) {
    std::unordered_map<AllocType, size_t> allocSizePerType;
    allocSizePerType[AllocType::InvListData] =
            calcVectorsEncodingMemorySpaceSize(
                    numVecs,
                    numSubQuantizers,
                    bitsPerSubQuantizer,
                    interleavedLayout);
    allocSizePerType[AllocType::InvListIndices] =
            calcIndicesMemorySpaceSize(numVecs, options);
    return allocSizePerType;
}

void IMIPQ::movePrecomputedCodesFrom(
        DeviceTensor<float, 3, true>& precomputedCode) {
    FAISS_ASSERT(precomputedCode.getSize(0) == quantizer_->getCodebookSize());
    FAISS_ASSERT(precomputedCode.getSize(1) == numSubQuantizers_);
    FAISS_ASSERT(precomputedCode.getSize(2) == numSubQuantizerCodes_);

    precomputedCodes_ = true;

    if (precomputedCode_.numElements() > 0) {
        precomputedCode_ = DeviceTensor<float, 3, true>();

    } else if (precomputedCodeHalf_.numElements() > 0) {
        precomputedCodeHalf_ = DeviceTensor<half, 3, true>();
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();
    if (useFloat16LookupTables_) {
        precomputedCodeHalf_ = DeviceTensor<half, 3, true>(
                resources_,
                makeDevAlloc(AllocType::QuantizerPrecomputedCodes, stream),
                {quantizer_->getCodebookSize(),
                 numSubQuantizers_,
                 numSubQuantizerCodes_});

        convertTensor(stream, precomputedCode, precomputedCodeHalf_);
    } else {
        precomputedCode_ = std::move(precomputedCode);
    }
}

void IMIPQ::setPrecomputedCodes(bool enable) {
    if (precomputedCodes_ != enable && !precomputeCodesOnCpu_) {
        precomputedCodes_ = enable;

        if (precomputedCodes_) {
            precomputeCodes_();
        } else {
            // Clear out old precomputed code data
            precomputedCode_ = DeviceTensor<float, 3, true>();
            precomputedCodeHalf_ = DeviceTensor<half, 3, true>();
        }
    }
}

void IMIPQ::appendVectors_(
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices,
        Tensor<int, 1, true>& uniqueLists,
        Tensor<int, 1, true>& vectorsByUniqueList,
        Tensor<int, 1, true>& uniqueListVectorStart,
        Tensor<int, 1, true>& uniqueListStartOffset,
        Tensor<ushort2, 1, true>& listIds,
        Tensor<int, 1, true>& listOffset,
        cudaStream_t stream) {
    FAISS_ASSERT(vecs.getSize(0) % quantizer_->getNumCodebooks() == 0);
    FAISS_ASSERT(vecs.getSize(1) * quantizer_->getNumCodebooks() == dim_);

    int vecsPerCodebook = vecs.getSize(0) / quantizer_->getNumCodebooks();
    FAISS_ASSERT(vecsPerCodebook == indices.getSize(0));

    //
    // Determine the encodings of the vectors
    //

    // For now we are restricted to <= 8 bits per code (hence uint8_t in the
    // encodings)
    FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);

    DeviceTensor<uint8_t, 2, true> encodings(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecsPerCodebook, numSubQuantizers_});

    {
        // Calculate the residual for each closest centroid
        DeviceTensor<float, 2, true> residuals(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {vecsPerCodebook, quantizer_->getDim()});

        quantizer_->computeResidual(vecs, listIds, residuals);

        // Residuals are in the form
        // (vec x numSubQuantizer x dimPerSubQuantizer)
        // transpose to
        // (numSubQuantizer x vec x dimPerSubQuantizer)
        auto residualsView = residuals.view<3>(
                {residuals.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});

        DeviceTensor<float, 3, true> residualsTranspose(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, residuals.getSize(0), dimPerSubQuantizer_});

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
                {numSubQuantizers_, residuals.getSize(0), 1});
        DeviceTensor<idx_t, 3, true> closestSubQIndex(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, residuals.getSize(0), 1});

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

        // The L2 distance function only returns idx_t indices. As we are
        // restricted to <= 8 bits per code, convert to uint8
        auto closestSubQIndex8 = convertTensorTemporary<idx_t, uint8_t, 3>(
                resources_, stream, closestSubQIndex);

        // Now, we have the nearest sub-q centroid for each slice of the
        // residual vector.
        auto closestSubQIndex8View = closestSubQIndex8.view<2>(
                {numSubQuantizers_, residuals.getSize(0)});

        // The encodings are finally a transpose of this data
        runTransposeAny(closestSubQIndex8View, 0, 1, encodings, stream);
    }

    DeviceTensor<unsigned int, 1, true> deviceListOffsetsTensor(
            deviceListOffsets_.data(), {(int)deviceListOffsets_.size()});

    // Append indices to the IVF lists
    if (indicesOptions_ == INDICES_64_BIT) {
        DeviceTensor<idx_t, 1, true> deviceListIndexTensor(
                (idx_t*)deviceListIndices_.data(),
                {(int)(deviceListIndices_.size() / sizeof(idx_t))});
        runIMIIndicesAppend(
                quantizer_->getCodebookSize(),
                listIds,
                listOffset,
                indices,
                indicesOptions_,
                deviceListIndexTensor,
                deviceListOffsetsTensor,
                stream);
    } else {
        DeviceTensor<int, 1, true> deviceListIndexTensor(
                (int*)deviceListIndices_.data(),
                {(int)(deviceListIndices_.size() / sizeof(int))});
        runIMIIndicesAppend(
                quantizer_->getCodebookSize(),
                listIds,
                listOffset,
                indices,
                indicesOptions_,
                deviceListIndexTensor,
                deviceListOffsetsTensor,
                stream);
    }

    DeviceTensor<uint8_t, 1, true, long> deviceListDataTensor(
            deviceListData_.data(), {(long)deviceListData_.size()});

    runIMIPQAppend(
            quantizer_->getCodebookSize(),
            listIds,
            listOffset,
            encodings,
            deviceListDataTensor,
            deviceListOffsetsTensor,
            numSubQuantizers_,
            stream);
}

size_t IMIPQ::getGpuVectorsEncodingSize_(int numVecs) const {
    return calcVectorsEncodingMemorySpaceSize(
            numVecs,
            numSubQuantizers_,
            bitsPerSubQuantizer_,
            interleavedLayout_);
}

size_t IMIPQ::getCpuVectorsEncodingSize_(int numVecs) const {
    size_t sizePerVector =
            utils::divUp(numSubQuantizers_ * bitsPerSubQuantizer_, 8);

    return (size_t)numVecs * sizePerVector;
}

// Convert the CPU layout to the GPU layout
std::vector<uint8_t> IMIPQ::translateCodesToGpu_(
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
std::vector<uint8_t> IMIPQ::translateCodesFromGpu_(
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

void IMIPQ::setPQCentroids_(float* data) {
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

void IMIPQ::precomputeCodesT_() {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    //
    //    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
    //        ---------------   ---------------------------       -------
    //            term 1                 term 2                   term 3
    //

    // Terms 1 and 3 are available only at query time. We compute term 2
    // here.

    // Compute 2 * (y_C|y_R) via batch matrix multiplication
    // batch size (sub q per c * codebook c) x {(centroid id)(sub dim) x (code
    // id)(sub dim)'}
    //         => (sub q) x {(centroid id)(sub dim) x (code id)(sub dim)'}
    //         => (sub q) x {(centroid id)(code id)}
    //         => (sub q)(centroid id)(code id)

    // View (centroid id)(dim) as
    //      (centroid id)(sub q)(dim)
    // Transpose (centroid id)(sub q)(sub dim) to
    //           (sub q)(centroid id)(sub dim)
    auto& coarseCentroids = quantizer_->getVectorsFloat32Ref();

    // Create the coarse PQ product
    DeviceTensor<float, 3, true> coarsePQProduct(
            resources_,
            makeTempAlloc(AllocType::QuantizerPrecomputedCodes, stream),
            {numSubQuantizers_,
             quantizer_->getCodebookSize(),
             numSubQuantizerCodes_});
    {
        int numSubQuantizersPerCodebook =
                numSubQuantizers_ / quantizer_->getNumCodebooks();

        auto centroidView = coarseCentroids.template view<3>(
                {coarseCentroids.getSize(0),
                 numSubQuantizersPerCodebook,
                 dimPerSubQuantizer_});

        // This is only needed temporarily
        DeviceTensor<float, 3, true> centroidsTransposed(
                resources_,
                makeTempAlloc(AllocType::QuantizerPrecomputedCodes, stream),
                {numSubQuantizersPerCodebook,
                 coarseCentroids.getSize(0),
                 dimPerSubQuantizer_});

        runTransposeAny(centroidView, 0, 1, centroidsTransposed, stream);

        auto centroidTransposedView = centroidsTransposed.template view<4>(
                {numSubQuantizersPerCodebook,
                 quantizer_->getNumCodebooks(),
                 quantizer_->getCodebookSize(),
                 dimPerSubQuantizer_});

        runIteratedMatrixMult<false>(
                coarsePQProduct,
                false,
                centroidTransposedView,
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
    DeviceTensor<float, 3, true> coarsePQProductTransposed(
            resources_,
            makeDevAlloc(AllocType::QuantizerPrecomputedCodes, stream),
            {quantizer_->getCodebookSize(),
             numSubQuantizers_,
             numSubQuantizerCodes_});
    runTransposeAny(coarsePQProduct, 0, 1, coarsePQProductTransposed, stream);

    // View (centroid id)(sub q)(code id) as
    //      (centroid id)(sub q * code id)
    auto coarsePQProductTransposedView = coarsePQProductTransposed.view<2>(
            {quantizer_->getCodebookSize(),
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
                {quantizer_->getCodebookSize(),
                 numSubQuantizers_,
                 numSubQuantizerCodes_});

        convertTensor(stream, coarsePQProductTransposed, precomputedCodeHalf_);
    } else {
        precomputedCode_ = std::move(coarsePQProductTransposed);
    }
}

void IMIPQ::precomputeCodes_() {
    FAISS_ASSERT(!quantizer_->getUseFloat16());
    precomputeCodesT_();
}

void IMIPQ::query(
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<long, 2, true>& outIndices) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    nprobe = std::min(nprobe, quantizer_->getSize());

    FAISS_ASSERT(queries.getSize(1) * quantizer_->getNumCodebooks() == dim_);
    FAISS_ASSERT(queries.getSize(0) % quantizer_->getNumCodebooks() == 0);
    FAISS_ASSERT(
            outDistances.getSize(0) * quantizer_->getNumCodebooks() ==
            queries.getSize(0));
    FAISS_ASSERT(outDistances.getSize(0) == outIndices.getSize(0));

    int numQueries = queries.getSize(0) / quantizer_->getNumCodebooks();

    // Reserve space for the closest coarse centroids
    DeviceTensor<float, 2, true> coarseDistances(
            resources_,
            makeTempAlloc(AllocType::CoarseDistancesOutput, stream),
            {numQueries, nprobe});
    DeviceTensor<ushort2, 2, true> coarseIndices(
            resources_,
            makeTempAlloc(AllocType::CoarseIndicesOutput, stream),
            {numQueries, nprobe});

    // Find the `nprobe` closest coarse centroids; we can use int
    // indices both internally and externally
    quantizer_->query(queries, nprobe, coarseDistances, coarseIndices, true);

    if (precomputedCodes_) {
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

void IMIPQ::query_split(
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<long, 2, true>& outIndices) {
    FAISS_ASSERT(queries.getSize(1) % quantizer_->getNumCodebooks() == 0);
    FAISS_ASSERT(
            queries.getSize(1) / quantizer_->getNumCodebooks() ==
            quantizer_->getSubDim());
    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto queriesView = queries.view<3>(
            {queries.getSize(0),
             quantizer_->getNumCodebooks(),
             quantizer_->getSubDim()});

    DeviceTensor<float, 3, true> queriesTranspose(
            resources_,
            makeTempAlloc(AllocType::QuerySplitted, stream),
            {quantizer_->getNumCodebooks(),
             queries.getSize(0),
             quantizer_->getSubDim()});

    runTransposeAny(queriesView, 0, 1, queriesTranspose, stream);

    auto queriesTransposeView = queriesTranspose.view<2>(
            {quantizer_->getNumCodebooks() * queries.getSize(0),
             quantizer_->getSubDim()});

    query(queriesTransposeView, nprobe, k, outDistances, outIndices);
}

int IMIPQ::getNumSubQuantizerCodes() {
    return numSubQuantizerCodes_;
}

Tensor<float, 3, true> IMIPQ::getPQCentroids() {
    return pqCentroidsMiddleCode_;
}

Tensor<float, 3, true> IMIPQ::getPrecomputedCodesVecFloat32() {
    return precomputedCode_;
}

void IMIPQ::runCalcTerm3(
        Tensor<float, 2, true>& queries,
        int& numQueries,
        int& numSubQuantizersPerCodebook,
        Tensor<float, 3, true>& term3) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto querySubQuantizerView = queries.view<3>(
            {queries.getSize(0),
             numSubQuantizersPerCodebook,
             dimPerSubQuantizer_});

    DeviceTensor<float, 3, true> queriesTransposed(
            resources_,
            makeTempAlloc(AllocType::QueryTransposed, stream),
            {numSubQuantizersPerCodebook,
             queries.getSize(0),
             dimPerSubQuantizer_});

    runTransposeAny(querySubQuantizerView, 0, 1, queriesTransposed, stream);

    auto queriesTransposedView = queriesTransposed.view<4>(
            {numSubQuantizersPerCodebook,
             quantizer_->getNumCodebooks(),
             numQueries,
             dimPerSubQuantizer_});

    DeviceTensor<float, 3, true> term3Transposed(
            resources_,
            makeTempAlloc(AllocType::Term3Transposed, stream),
            {numSubQuantizers_, numQueries, numSubQuantizerCodes_});

    runIteratedMatrixMult<false>(
            term3Transposed,
            false,
            queriesTransposedView,
            false,
            pqCentroidsMiddleCode_,
            true,
            -2.0f,
            0.0f,
            resources_->getBlasHandleCurrentDevice(),
            stream);

    runTransposeAny(term3Transposed, 0, 1, term3, stream);
}

void IMIPQ::runCalcTerm3(
        Tensor<float, 2, true>& queries,
        Tensor<float, 3, true>& term3) {
    int numQueries = queries.getSize(0) / quantizer_->getNumCodebooks();
    int numSubQuantizersPerCodebook =
            numSubQuantizers_ / quantizer_->getNumCodebooks();

    runCalcTerm3(queries, numQueries, numSubQuantizersPerCodebook, term3);
}

void IMIPQ::runPQPrecomputedCodes_(
        Tensor<float, 2, true>& queries,
        DeviceTensor<float, 2, true>& coarseDistances,
        DeviceTensor<ushort2, 2, true>& coarseIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<long, 2, true>& outIndices) {
    FAISS_ASSERT(precomputedCode_.numElements() > 0);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    int numQueries = queries.getSize(0) / quantizer_->getNumCodebooks();

    // Compute precomputed code term 3, - 2 * (x|y_R)
    // This is done via batch MM
    // {sub q} x {(query id)(sub dim) * (code id)(sub dim)'} =>
    // {sub q} x {(query id)(code id)}
    DeviceTensor<float, 3, true> term3Transposed(
            resources_,
            makeTempAlloc(AllocType::Term3, stream),
            {numQueries, numSubQuantizers_, numSubQuantizerCodes_});

    int numSubQuantizersPerCodebook =
            numSubQuantizers_ / quantizer_->getNumCodebooks();

    runCalcTerm3(
            queries, numQueries, numSubQuantizersPerCodebook, term3Transposed);

    auto term3TransposedView = term3Transposed.view<4>(
            {numQueries,
             quantizer_->getNumCodebooks(),
             numSubQuantizersPerCodebook,
             numSubQuantizerCodes_});

    NoTypeTensor<4, true> term2;
    NoTypeTensor<4, true> term3;
    DeviceTensor<half, 4, true> term3Half;

    if (useFloat16LookupTables_) {
        auto precomputedCodeViewHalf = precomputedCodeHalf_.view<4>(
                {precomputedCodeHalf_.getSize(0),
                 quantizer_->getNumCodebooks(),
                 numSubQuantizersPerCodebook,
                 precomputedCodeHalf_.getSize(2)});
        term3Half = convertTensorTemporary<float, half, 4>(
                resources_, stream, term3TransposedView);
        term2 = NoTypeTensor<4, true>(precomputedCodeViewHalf);
        term3 = NoTypeTensor<4, true>(term3Half);
    } else {
        auto precomputedCodeView = precomputedCode_.view<4>(
                {precomputedCode_.getSize(0),
                 quantizer_->getNumCodebooks(),
                 numSubQuantizersPerCodebook,
                 precomputedCode_.getSize(2)});
        term2 = NoTypeTensor<4, true>(precomputedCodeView);
        term3 = NoTypeTensor<4, true>(term3TransposedView);
    }

    DeviceTensor<unsigned int, 1, true> deviceListOffsetsTensor(
            deviceListOffsets_.data(), {(int)deviceListOffsets_.size()});
    DeviceTensor<uint8_t, 1, true, long> deviceListDataTensor(
            deviceListData_.data(), {(long)deviceListData_.size()});

    if (indicesOptions_ == INDICES_64_BIT) {
        DeviceTensor<idx_t, 1, true> deviceListIndexTensor(
                (idx_t*)deviceListIndices_.data(),
                {(int)(deviceListIndices_.size() / sizeof(idx_t))});

        runPQScanMultiPassPrecomputed(
                coarseDistances, // term 1
                term2,           // term 2
                term3,           // term 3
                quantizer_->getCodebookSize(),
                coarseIndices,
                useFloat16LookupTables_,
                interleavedLayout_,
                bitsPerSubQuantizer_,
                numSubQuantizers_,
                numSubQuantizerCodes_,
                deviceListOffsetsTensor,
                deviceListDataTensor,
                numSubQuantizers_,
                deviceListIndexTensor,
                indicesOptions_,
                maxListLength_,
                k,
                outDistances,
                outIndices,
                resources_);
    } else {
        DeviceTensor<int, 1, true> deviceListIndexTensor(
                (int*)deviceListIndices_.data(),
                {(int)(deviceListIndices_.size() / sizeof(int))});

        runPQScanMultiPassPrecomputed(
                coarseDistances, // term 1
                term2,           // term 2
                term3,           // term 3
                quantizer_->getCodebookSize(),
                coarseIndices,
                useFloat16LookupTables_,
                interleavedLayout_,
                bitsPerSubQuantizer_,
                numSubQuantizers_,
                numSubQuantizerCodes_,
                deviceListOffsetsTensor,
                deviceListDataTensor,
                numSubQuantizers_,
                deviceListIndexTensor,
                indicesOptions_,
                maxListLength_,
                k,
                outDistances,
                outIndices,
                resources_);
    }
}

template <typename CentroidT>
void IMIPQ::runPQNoPrecomputedCodesT_(
        Tensor<float, 2, true>& queries,
        DeviceTensor<float, 2, true>& coarseDistances,
        DeviceTensor<ushort2, 2, true>& coarseIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<long, 2, true>& outIndices) {}

void IMIPQ::runPQNoPrecomputedCodes_(
        Tensor<float, 2, true>& queries,
        DeviceTensor<float, 2, true>& coarseDistances,
        DeviceTensor<ushort2, 2, true>& coarseIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<long, 2, true>& outIndices) {
    if (quantizer_->getUseFloat16()) {
        runPQNoPrecomputedCodesT_<half>(
                queries,
                coarseDistances,
                coarseIndices,
                k,
                outDistances,
                outIndices);
    } else {
        runPQNoPrecomputedCodesT_<float>(
                queries,
                coarseDistances,
                coarseIndices,
                k,
                outDistances,
                outIndices);
    }
}

} // namespace gpu
} // namespace faiss
