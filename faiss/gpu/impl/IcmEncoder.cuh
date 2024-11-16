/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
    int M;    ///< number of codebooks
    int K;    ///< number of codewords in a codebook
    int dims; ///< dimensions of a codeword

    GpuResourcesProvider* prov;
    std::shared_ptr<GpuResources> res;
    int device;

    DeviceTensor<float, 4, true> bterm;     ///< bianry terms, size [M, M, K, K]
    DeviceTensor<float, 3, true> codebooks; ///< codebooks, size [M, K, dims]

    IcmEncoderImpl(
            int M,
            int K,
            int dims,
            GpuResourcesProvider* prov,
            int device);

    ~IcmEncoderImpl() {}

    ///< copy codebooks to device memory and compute unary terms
    void setBinaryTerm(const float* codebooks);

    /** Compute unary terms.
     *
     * uterm[i] = x * codebook[i]^T, i = 1,...,M
     *
     * @param uterm     output unary terms, size [M, n, K]
     * @param x         input vectors, size [n, dims]
     * @param codebooks codebooks, size [M, K, dims]
     * @param n         number of input vectors
     */
    void computeUnaryTerms(
            float* bterm,
            const float* x,
            const float* codebooks,
            int n) const;

    /** Compute binary terms.
     *
     * bterm[i][j] = codebooks[i] * codebooks[j]^T. i, j = 1,...,M
     *
     * @param bterm     output binary terms, size [M, M, K, K]
     * @param codebooks codebooks, size [M, K, dims]
     */
    void computeBinaryTerms(float* bterm, const float* codebooks) const;

    ///< icm encode method
    void encode(
            int32_t* codes,
            const float* x,
            const float* codebooks,
            std::mt19937& gen,
            int n,
            int nperts,
            int ilsIters,
            int icmIters) const;
};

} // namespace gpu
} // namespace faiss
