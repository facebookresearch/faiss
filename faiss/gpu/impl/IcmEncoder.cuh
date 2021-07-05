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
    DeviceTensor<float, 3, true> uterm;
    DeviceTensor<float, 4, true> bterm;
    GpuResourcesProvider* prov;
    std::shared_ptr<GpuResources> res;

    IcmEncoderImpl(int M, int K, GpuResourcesProvider* prov);

    void set_unary_term(int n, const float* unaries);

    void set_binary_term(const float* binaries);

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
