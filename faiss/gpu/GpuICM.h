/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/LocalSearchQuantizer.h>

namespace faiss {
namespace gpu {

class GpuResourcesProvider;
struct IcmEncoder;

class GpuLSQIcmEncoder : public LSQIcmEncoder {
public:
    GpuLSQIcmEncoder(GpuResourcesProvider* prov);

    ~GpuLSQIcmEncoder();

    void set_unary_term(size_t n, const float* unaries) override;

    void set_binary_term(const float* binaries) override;

    void encode(int32_t* codes, size_t n) const override;

private:
    IcmEncoder* encoder;
};

} // namespace gpu
} // namespace faiss
