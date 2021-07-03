/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuLSQIcmEncoder.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/ICM.cuh>

namespace faiss {
namespace gpu {

GpuLSQIcmEncoder::GpuLSQIcmEncoder(
        size_t M,
        size_t K,
        GpuResourcesProvider* prov)
        : LSQIcmEncoder(M, K), encoder(new IcmEncoder(M, K, prov)) {}

void GpuLSQIcmEncoder::set_unary_term(size_t n, const float* unaries) {
    encoder->set_unary_term(n, unaries);
}

void GpuLSQIcmEncoder::set_binary_term(const float* binaries) {
    encoder->set_binary_term(binaries);
}

void GpuLSQIcmEncoder::encode(int32_t* codes, size_t n) const {
    encoder->encode(codes, n);
}

GpuLSQIcmEncoderFactory::GpuLSQIcmEncoderFactory() {
    prov = new StandardGpuResources();
}

LSQIcmEncoder* GpuLSQIcmEncoderFactory::get(size_t M, size_t K) {
    return new GpuLSQIcmEncoder(M, K, prov);
}

} // namespace gpu
} // namespace faiss
