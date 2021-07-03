/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
namespace gpu {

struct IcmEncoder {
    size_t M, K;
    DeviceTensor<float, 3, true> uterm;
    DeviceTensor<float, 4, true> bterm;
    GpuResourcesProvider* prov;
    std::shared_ptr<GpuResources> res;

    IcmEncoder(size_t M, size_t K, GpuResourcesProvider* prov);

    void set_unary_term(size_t n, const float* unaries);

    void set_binary_term(const float* binaries);

    void encode(int32_t* codes, size_t n) const;

    template <size_t kNumCodebooks>
    void encode_dispatch_k(int32_t* codes, size_t n, size_t K) const;

    template <size_t kNumCodebooks, size_t kCodebookSize>
    void encode_impl(int32_t* codes_host, size_t n) const;
};

} // namespace gpu
} // namespace faiss
