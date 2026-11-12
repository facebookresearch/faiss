// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal IVF PQ implementation: GPU-resident IVF list storage for
 * 8-bit product-quantized codes. Each vector is M bytes (one byte
 * per subquantizer, ksub=256).
 */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/gpu_metal/MetalResources.h>

namespace faiss {
namespace gpu_metal {

class MetalIVFPQImpl {
   public:
    MetalIVFPQImpl(
            std::shared_ptr<MetalResources> resources,
            int dim,
            idx_t nlist,
            int numSubQuantizers,
            int bitsPerCode,
            faiss::MetricType metric,
            float metricArg);

    ~MetalIVFPQImpl();

    void reset();
    void reserveMemory(idx_t totalVecs);

    /// Append PQ-encoded vectors. codes: n * codeSize bytes.
    void appendCodes(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            const idx_t* xids);

    /// Upload PQ centroids: M * ksub * dsub floats, row-major.
    /// Layout: pqCentroids[m][c][dsub_dim]
    void setPQCentroids(const float* centroids);

    int dim() const {
        return dim_;
    }
    idx_t nlist() const {
        return nlist_;
    }
    int numSubQuantizers() const {
        return M_;
    }
    int bitsPerCode() const {
        return bitsPerCode_;
    }
    int ksub() const {
        return ksub_;
    }
    int dsub() const {
        return dsub_;
    }
    size_t codeSize() const {
        return codeSize_;
    }

    const std::vector<size_t>& listLength() const {
        return listLength_;
    }
    const std::vector<size_t>& listOffset() const {
        return listOffset_;
    }

    id<MTLBuffer> codesBuffer() const {
        return codesBuffer_;
    }
    id<MTLBuffer> idsBuffer() const {
        return idsBuffer_;
    }
    id<MTLBuffer> listOffsetGpuBuffer() const {
        return listOffsetBuf_;
    }
    id<MTLBuffer> listLengthGpuBuffer() const {
        return listLengthBuf_;
    }
    id<MTLBuffer> pqCentroidsBuffer() const {
        return pqCentroidsBuf_;
    }
    size_t totalVecs() const {
        return totalVecs_;
    }

   private:
    void uploadToGpu();

    std::shared_ptr<MetalResources> resources_;

    int dim_;
    idx_t nlist_;
    int M_;
    int bitsPerCode_;
    int ksub_;
    int dsub_;
    size_t codeSize_;
    faiss::MetricType metric_type_;
    float metric_arg_;

    std::vector<size_t> listLength_;
    std::vector<size_t> listOffset_;

    std::vector<uint8_t> hostCodes_;
    std::vector<idx_t> hostIds_;
    size_t totalVecs_;

    id<MTLBuffer> codesBuffer_;
    id<MTLBuffer> idsBuffer_;
    id<MTLBuffer> listOffsetBuf_;
    id<MTLBuffer> listLengthBuf_;
    id<MTLBuffer> pqCentroidsBuf_;
};

} // namespace gpu_metal
} // namespace faiss
