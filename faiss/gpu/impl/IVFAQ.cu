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
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAQ.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
//#include <faiss/gpu/impl/PQCodeDistances.cuh>
//#include <faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh>
//#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
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

IVFAQ::IVFAQ(
        GpuResources* resources,
        faiss::MetricType metric,
        float metricArg,
        FlatIndex* quantizer,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useFloat16LookupTables,
        bool useMMCodeDistance,
        bool interleavedLayout,
        float* aqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFBase(resources,
                  metric,
                  metricArg,
                  quantizer,
                  interleavedLayout,
                  indicesOptions,
                  space),
          numSubQuantizers_(numSubQuantizers),
          // bitsPerSubQuantizer_(bitsPerSubQuantizer),
          numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
          dimPerSubQuantizer_(dim_ / numSubQuantizers),
          useFloat16LookupTables_(useFloat16LookupTables),
          useMMCodeDistance_(useMMCodeDistance),
          precomputedCodes_(false) {
    FAISS_ASSERT(aqCentroidData);

    // FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
    FAISS_ASSERT(dim_ % numSubQuantizers_ == 0);
    FAISS_ASSERT(
            interleavedLayout || isSupportedAQCodeLength(numSubQuantizers_));

    setAQCentroids_(aqCentroidData);
}

IVFAQ::~IVFAQ() {}

void IVFAQ::setAQCentroids_(float* data) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    size_t aqSize = numSub;

    // Make sure the data is on the host
    thrust::host_vector<float> hostMemory;
    hostMemory.insert(hostMemory.end(), data, data + aqSize);

    HostTensor<float, 3, true> aqHost(hostMemory.data(), {});
}

void IVFAQ::setPrecomputedCodes(bool enable) {
    if (enable && metric_ == MetricType::METRIC_INNER_PRODUCT) {
        fprintf(stderr,
                "Precomputed codes are not needed for GpuIndexIVFPQ "
                "with METRIC_INNER_PRODUCT");
        return;
    }

    if (precomputedCodes_ != enable) {
        precomputedCodes_ = enable;

        if (precomputedCodes_) {
            precomputedCodes_();
        } else {
            // Clear out old precomputed code data
            precomputedCode_ = DeviceTensor<float, 3, true>();
            precomputedCodeHalf_ = DeviceTensor<half, 3, true>();
        }
    }
}

void IVFAQ::appendVectors_() {
    // Determine the encodings of the vectors

    FAISS_ASSERT(bitsPerSubQuantizer_ <= 8);
    DeviceTensor<uint8_t, 2, true> encodings(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), numSubQuantizers_});

    {
        // Calculate the residual for each closest centroid
        DeviceTensor<float, 2, true> residuals(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {vecs.getSize(0), vecs.getSize(1)});

        if (quantizer_->getUseFloat16()) {
            auto& coarseCentroids = quantizer_->getVectorsFloat16Ref();
            runCalcResidual(vecs, coarseCentroids, listIds, residuals, stream);
        } else {
            auto& coarseCentroids = quantizer_->getVectorsFloat32Ref();
            runCalcResidual(vecs, coarseCentroids, listIds, residuals, stream);
        }
        // Residuals are in the form
        // (vec x numSubQuantizer x dimPerSubQuantizer)
        // transpose to
        // (numSubQuantizer x vec x dimPerSubQuantizer)
        auto residualView = residuals.view<3>(
                {residuals.getSize(0), numSubQuantizers_, dimPerSubQuantizer_});

        DeviceTensor<float, 3, true> residualsTranspose(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, residuals.getSize(0), dimPerSubQuantizer_});
        runTransposeAny(residualsView, 0, 1, residualsTranspose, stream);

        // batch op to find the top-1 distances:
        // batch size: numSubQuantizer
        // centroids: (numSubQuantizerCodes x dimPerSubQuantizer)
        // residuals: (vec x dimPerSubQuantizer)
        // ==> (numSubQuantizer x vec x 1)
        DeviceTensor<float, 3, true> closestSubQDistance(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, residuals.getSize(0), 1});
        DeviceTensor<int, 3, true> closestSubQIndex(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                {numSubQuantizers_, residuals.getSize(0), 1});
        for (int subQ = 0; subQ < numSubQuantizers_; ++subQ) {
            auto closestSubQDistanceView = closestSubQDistance[subQ].view();
            auto closestSubQIndexView = closestSubQIndex[subQ].view();

            auto aqCentroidsMiddleCodeView =
                    aqCentroidsMiddleCode_[subQ].view();
            auto residualsTransposeView = residualsTranspose[subQ].view();

            runL2Distance(
                    resources_,
                    stream,
                    aqCentroidsMiddleCodeView,
                    true,    // row major
                    nullptr, // no precomputed norms
                    residualsTransposeView,
                    true,
                    1,
                    closestSubQDistanceView,
                    closestSubQIndexView,
                    true);
        }
        // convert int32 indices to uint8
        auto closestSubQIndex8 = convertTensorTemporary<int, uint8_t, 3>(
                resources_, stream, closestSubQindex);

        auto closestSubQindex8View = closestSubQindex8.view<2>(
                {numSubQuantizers_, residuals.getSize(0)});

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
        // reuse the PQ append impl
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
        );
    } else {
        runIVFPQAppend(
                listIds,
                listOffeset,
                encodings,
                deviceListDataPointers_,
                stream);
    }
}

size_t IVFAQ::getGpuVectorsEncodingSize_(int numVecs) const {
    if (interleavedLayout_) {
        // bits per AQ code
        int bits = bitsPerSubQuantizer_;

        // bytes to encode a block of 32 vectors (single AQ code)
        int bytesPerDimBlock = bits * 32 / 8;

        // bytes to fully encode 32 vectors
        int bytesPerBlock = bytesPerDimBlock * numSubQuantizers_;

        // number of blocks of 32 vectors we have
    }
}

void IVFPQ::query(
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,

) {}

} // namespace gpu
} // namespace faiss
