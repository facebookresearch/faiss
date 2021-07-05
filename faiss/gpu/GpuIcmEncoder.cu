/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIcmEncoder.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IcmEncoder.cuh>

namespace faiss {
namespace gpu {

GpuIcmEncoder::GpuIcmEncoder(
        size_t M,
        size_t K,
        GpuResourcesProvider* prov)
        : lsq::IcmEncoder(M, K), encoder(new IcmEncoderImpl(M, K, prov)) {}

void GpuIcmEncoder::set_unary_term(size_t n, const float* unaries) {
    encoder->setUnaryTerm(n, unaries);
}

void GpuIcmEncoder::set_binary_term(const float* binaries) {
    encoder->setBinaryTerm(binaries);
}

void GpuIcmEncoder::encode(
    const float* x,
    const float* codebooks,
    int32_t* codes,
    std::mt19937& gen,
    size_t n,
    size_t d,
    size_t nperts,
    size_t ils_iters,
    size_t icm_iters) const {
    encoder->encode(x, codebooks, codes, gen, n, d, nperts, ils_iters, icm_iters);
}

GpuIcmEncoderFactory::GpuIcmEncoderFactory() {
    prov = new StandardGpuResources();
}

lsq::IcmEncoder* GpuIcmEncoderFactory::get(size_t M, size_t K) {
    return new GpuIcmEncoder(M, K, prov);
}

} // namespace gpu
} // namespace faiss
