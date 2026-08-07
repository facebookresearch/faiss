// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MetalKernels: typed wrapper around Metal compute kernels.
 * Owns library compilation, pipeline caching, and dispatch encoding.
 */

#pragma once

#import <Metal/Metal.h>
#include <faiss/MetricType.h>
#include <string>
#include <unordered_map>

namespace faiss {
namespace gpu_metal {

enum class IVFScanVariant { Standard, Small, Interleaved };

class MetalKernels {
   public:
    explicit MetalKernels(id<MTLDevice> device);
    ~MetalKernels();

    bool isValid() const;
    static constexpr int kMaxK = 2048;

    void encodeDistanceMatrix(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            int nq,
            int nb,
            int d,
            MetricType metric);

    void encodeL2WithNorms(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> distances,
            id<MTLBuffer> vecNorms,
            int nq,
            int nb,
            int d);

    void encodeComputeNorms(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> vectors,
            id<MTLBuffer> norms,
            int nb,
            int d);

    void encodeTopKThreadgroup(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> distances,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            int nq,
            int nb,
            int k,
            bool wantMin);

    void encodeIVFScanList(
            id<MTLComputeCommandEncoder> enc,
            IVFScanVariant variant,
            id<MTLBuffer> queries,
            id<MTLBuffer> codes,
            id<MTLBuffer> ids,
            id<MTLBuffer> listOffset,
            id<MTLBuffer> listLength,
            id<MTLBuffer> coarseAssign,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            id<MTLBuffer> paramsBuf,
            int nq,
            int nprobe,
            id<MTLBuffer> ilCodesOffset = nil);

    void encodeIVFMergeLists(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            id<MTLBuffer> paramsBuf,
            int nq);

    // ---- IVF-PQ ----

    void encodeIVFPQBuildLookupTables(
            id<MTLComputeCommandEncoder> enc,
            bool isL2,
            bool outFp16,
            id<MTLBuffer> queries,
            id<MTLBuffer> coarseAssign,
            id<MTLBuffer> coarseCentroids,
            id<MTLBuffer> pqCentroids,
            id<MTLBuffer> outLookup,
            int nq,
            int d,
            int M,
            int nprobe);

    void encodeIVFPQScanList(
            id<MTLComputeCommandEncoder> enc,
            bool useSmall,
            bool useFp16Lookup,
            id<MTLBuffer> lookupTable,
            id<MTLBuffer> codes,
            id<MTLBuffer> ids,
            id<MTLBuffer> listOffset,
            id<MTLBuffer> listLength,
            id<MTLBuffer> coarseAssign,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            id<MTLBuffer> paramsBuf,
            int nq,
            int nprobe);

    // ---- IVF-PQ precomputed-table path ----

    void encodeIVFPQPrecomputeTerm2(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> coarseCentroids,
            id<MTLBuffer> pqCentroids,
            id<MTLBuffer> outTerm2,
            int nlist,
            int d,
            int M);

    void encodeIVFPQBuildQueryTerm(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> queries,
            id<MTLBuffer> pqCentroids,
            id<MTLBuffer> outQTerm,
            int nq,
            int d,
            int M,
            bool isL2);

    void encodeIVFPQScanListPrecomp(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> term2, // nil when useTerm2 is false (IP path)
            id<MTLBuffer> qterm,
            id<MTLBuffer> coarseDist,
            id<MTLBuffer> codes,
            id<MTLBuffer> ids,
            id<MTLBuffer> listOffset,
            id<MTLBuffer> listLength,
            id<MTLBuffer> coarseAssign,
            id<MTLBuffer> perListDist,
            id<MTLBuffer> perListIdx,
            int nq,
            int M,
            int k,
            int nprobe,
            bool wantMin,
            bool useTerm2,
            bool useDis0);

    void encodeIVFMergeListsGrouped(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> inDist,
            id<MTLBuffer> inIdx,
            id<MTLBuffer> outDist,
            id<MTLBuffer> outIdx,
            int nq,
            int numLists,
            int groupSize,
            int k,
            bool wantMin);

    void encodeConvertF32ToF16(
            id<MTLComputeCommandEncoder> enc,
            id<MTLBuffer> src,
            id<MTLBuffer> dst,
            size_t numElems);

    static int selectTopKVariantIndex(int k);

   private:
    id<MTLComputePipelineState> pipeline(const char* name);

    id<MTLDevice> device_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache_;

    static constexpr int kTopKVariantSizes[] =
            {32, 64, 128, 256, 512, 1024, 2048};
    static constexpr int kNumTopKVariants = 7;
};

MetalKernels& getMetalKernels(id<MTLDevice> device);

} // namespace gpu_metal
} // namespace faiss
