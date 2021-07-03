/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuICM.h>
#include <faiss/gpu/impl/ICM.cuh>
#include <faiss/gpu/GpuResources.h>

namespace faiss {
namespace gpu {

GpuLSQIcmEncoder::GpuLSQIcmEncoder(GpuResourcesProvider* prov) : encoder(new IcmEncoder(prov)) {}

GpuLSQIcmEncoder::~GpuLSQIcmEncoder() { delete encoder; }

void GpuLSQIcmEncoder::set_unary_term(size_t n, const float* unaries) {
    encoder->set_unary_term(n, unaries);
}

void GpuLSQIcmEncoder::set_binary_term(const float* binaries) {
    encoder->set_binary_term(binaries);
}

void GpuLSQIcmEncoder::encode(int32_t* codes, size_t n) const {
    encoder->encode(codes, n);
}

} // namespace gpu
} // namespace faiss
 