/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>

#include <random>

namespace faiss {
namespace gpu {

struct IcmEncoderImpl {
    int M, K;
    GpuResourcesProvider* prov;
    std::shared_ptr<GpuResources> res;
    int device;
    DeviceTensor<float, 4, true> bterm;
    DeviceTensor<float, 3, true> codebooks;

    IcmEncoderImpl(int M, int K, GpuResourcesProvider* prov, int device);

    ~IcmEncoderImpl() {}

    void setBinaryTerm(const float* codebooks, int dims);

    void computeUnaryTerms(
            float* bterm,
            const float* x,
            const float* codebooks,
            int n,
            int dims) const;

    void computeBinaryTerms(
            float* bterm,
            const float* codebooks,
            int dims) const;

    template <int K>
    void encodeImpl(
            const float* x,
            const float* codebooks,
            int32_t* codes,
            std::mt19937& gen,
            int n,
            int dims,
            int nperts,
            int ilsIters,
            int icmIters) const;

    void encode(
            const float* x,
            const float* codebooks,
            int32_t* codes,
            std::mt19937& gen,
            int n,
            int dims,
            int nperts,
            int ilsIters,
            int icmIters) const;
};

} // namespace gpu
} // namespace faiss
